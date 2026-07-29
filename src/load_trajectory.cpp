#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "load_trajectory.hpp"

// ── CSV I/O ────────────────────────────────────────────────────────────────
std::vector<std::array<double, 7>> loadTrajectoryCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::vector<std::array<double, 7>> traj;
    std::string line;

    while (std::getline(f, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        std::array<double, 7> wp;
        std::stringstream ss(line);
        std::string token;
        int j = 0;

        while (std::getline(ss, token, ',') && j < 7) {
            try {
                wp[j++] = std::stod(token);
            } catch (const std::exception&) {
                throw std::runtime_error("Bad value at line: " + line);
            }
        }

        if (j != 7)
            throw std::runtime_error("Expected 7 values, got "
                                     + std::to_string(j)
                                     + " at line: " + line);
        traj.push_back(wp);
    }

    std::cout << "Loaded " << traj.size() << " waypoints from " << path << "\n";
    return traj;
}

void saveTrajectoryCSV(const std::string& path,
                       const std::vector<std::array<double, 7>>& traj) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << std::fixed << std::setprecision(9);
    f << "# q1,q2,q3,q4,q5,q6,q7\n";
    for (const auto& wp : traj) {
        for (int j = 0; j < 7; ++j) {
            f << wp[j];
            if (j < 6) f << ",";
        }
        f << "\n";
    }

    std::cout << "Saved " << traj.size() << " waypoints to " << path << "\n";
}

std::vector<double> loadTimestampsCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::vector<double> t;
    std::string line;

    while (std::getline(f, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        double wp;
        std::stringstream ss(line);
        std::string token;
        int j = 0;

        while (std::getline(ss, token, ',') && j++ < 1) {
            try {
                wp = std::stod(token);
            } catch (const std::exception&) {
                throw std::runtime_error("Bad value at line: " + line);
            }
        }

        if (j != 1)
            throw std::runtime_error("Expected 1 value, got "
                                     + std::to_string(j)
                                     + " at line: " + line);
        t.push_back(wp);
    }

    std::cout << "Loaded " << t.size() << " waypoints from " << path << "\n";
    return t;
}

void saveTimestampsCSV(const std::string& path,
                       const std::vector<double>& times) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    f << std::fixed << std::setprecision(9);
    f << "# t\n";
    for (const double t : times)
        f << t << "\n";

    std::cout << "Saved " << times.size() << " timestamps to " << path << "\n";
}

// ── Binary I/O ────────────────────────────────────────────────────────────
std::vector<std::array<double, 7>> loadBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    size_t n = f.tellg() / (7 * sizeof(double));
    f.seekg(0);

    std::vector<std::array<double, 7>> traj(n);
    for (auto& wp : traj)
        f.read(reinterpret_cast<char*>(wp.data()), 7 * sizeof(double));
    return traj;
}

void saveBin(const std::string& path,
             const std::vector<std::array<double, 7>>& traj) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    for (const auto& wp : traj)
        f.write(reinterpret_cast<const char*>(wp.data()), 7 * sizeof(double));
}

// ── Finite-difference velocity estimation ──────────────────────────────────
// Central differences for interior points, one-sided for endpoints
Eigen::VectorXd estimateVelocities(const Eigen::VectorXd& t,
                                    const Eigen::VectorXd& q) {
    int n = q.size();
    Eigen::VectorXd v(n);

    // Endpoints: one-sided (clamped to zero for start/stop motions)
    v(0)     = 0.0;
    v(n - 1) = 0.0;

    // Interior: central differences
    for (int i = 1; i < n - 1; ++i)
        v(i) = (q(i + 1) - q(i - 1)) / (t(i + 1) - t(i - 1));

    return v;
}

// ── Finite-difference acceleration estimation ───────────────────────────────
Eigen::VectorXd estimateAccelerations(const Eigen::VectorXd& t,
                                       const Eigen::VectorXd& q) {
    int n = q.size();
    Eigen::VectorXd a(n);

    // Endpoints: clamped to zero (robot starts and ends at rest)
    a(0)     = 0.0;
    a(n - 1) = 0.0;

    // Interior: central second differences
    for (int i = 1; i < n - 1; ++i) {
        double h0 = t(i)     - t(i - 1);
        double h1 = t(i + 1) - t(i);
        a(i) = 2.0 * ((q(i + 1) - q(i)) / h1 - (q(i) - q(i - 1)) / h0)
                   / (h0 + h1);
    }

    return a;
}

// ── Per-joint quintic spline interpolation ──────────────────────────────────
Eigen::VectorXd quinticSplineInterp(const Eigen::VectorXd& t_low,
                                     const Eigen::VectorXd& q_low,
                                     const Eigen::VectorXd& t_high) {
    int n      = t_low.size();
    int m      = t_high.size();

    // Estimate derivatives at knots
    Eigen::VectorXd v_low = estimateVelocities(t_low, q_low);
    Eigen::VectorXd a_low = estimateAccelerations(t_low, q_low);

    Eigen::VectorXd q_high(m);
    int seg = 0;

    for (int i = 0; i < m; ++i) {
        double t = t_high(i);

        // Advance segment pointer
        while (seg < n - 2 && t > t_low(seg + 1)) ++seg;

        double h = t_low(seg + 1) - t_low(seg);
        double s = (t - t_low(seg)) / h;  // normalize to [0, 1]

        q_high(i) = quinticHermite(s, h,
                                    q_low(seg),   v_low(seg),   a_low(seg),
                                    q_low(seg+1), v_low(seg+1), a_low(seg+1));
    }

    return q_high;
}

// ── Main interpolation entry point ──────────────────────────────────────────
std::vector<std::array<double, 7>> interpolateTo1kHz(
        const std::vector<std::array<double, 7>>& traj_low,
        std::vector<double> time_low) {

    int    n        = traj_low.size();
    double duration = time_low[n-1]; // (n - 1) / source_rate_hz;
    int    n_high   = static_cast<int>(duration * 1000.0);

    std::cout << "Interpolating from " << n << " waypoints to "
              << n_high << " waypoints over " << duration << " seconds\n";

    // Time axes
    Eigen::VectorXd t_low(n), t_high(n_high);
    for (int i = 0; i < n;     ++i) t_low(i)  = time_low[i];
    for (int i = 0; i < n_high; ++i) t_high(i) = i / 1000.0;

    // Interpolate each of the 7 joints independently
    std::vector<std::array<double, 7>> traj_high(n_high);

    for (int j = 0; j < 7; ++j) {
        Eigen::VectorXd q_low(n);
        for (int i = 0; i < n; ++i)
            q_low(i) = traj_low[i][j];

        Eigen::VectorXd q_high = quinticSplineInterp(t_low, q_low, t_high);

        for (int i = 0; i < n_high; ++i)
            traj_high[i][j] = q_high(i);
    }

    return traj_high;
}

// ── Sanity checks ────────────────────────────────────────────────────────
// Note: default value for rate_hz is declared in the header only —
// repeating it here would be a compile error.
void validateTrajectory(const std::vector<std::array<double, 7>>& traj,
                         double rate_hz) {
    // Franka Panda limits
    constexpr double VEL_LIMIT  = 2.175;    // rad/s
    constexpr double ACC_LIMIT  = 15.0;     // rad/s²
    constexpr double STEP_LIMIT = 0.01;     // rad per 1ms tick

    double max_step = 0.0, max_vel = 0.0, max_acc = 0.0;
    double dt = 1.0 / rate_hz;

    for (size_t i = 1; i < traj.size(); ++i) {
        for (int j = 0; j < 7; ++j) {
            double dp = std::abs(traj[i][j] - traj[i-1][j]);
            double v  = dp / dt;
            max_step = std::max(max_step, dp);
            max_vel  = std::max(max_vel,  v);

            if (i >= 2) {
                double dp_prev = std::abs(traj[i-1][j] - traj[i-2][j]);
                double dv = std::abs(v - dp_prev / dt);
                max_acc = std::max(max_acc, dv / dt);
            }
        }
    }

    std::cout << "── Trajectory validation ──────────────────────\n";
    std::cout << "Waypoints  : " << traj.size() << "\n";
    std::cout << "Max step   : " << max_step << " rad  "
              << (max_step > STEP_LIMIT ? "⚠ HIGH" : "✓ OK") << "\n";
    std::cout << "Max vel    : " << max_vel  << " rad/s  "
              << (max_vel  > VEL_LIMIT  ? "⚠ EXCEEDS LIMIT" : "✓ OK") << "\n";
    std::cout << "Max acc    : " << max_acc  << " rad/s²  "
              << (max_acc  > ACC_LIMIT  ? "⚠ EXCEEDS LIMIT" : "✓ OK") << "\n";
    std::cout << "───────────────────────────────────────────────\n";
}