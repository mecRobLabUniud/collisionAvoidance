#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include <franka/robot.h>
#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/robot_state.h>
#include <franka/exception.h>

#include "examples_common.hpp"

// ---------------------------------------------------------------------------
// 1. Load trajectory from file
//    Expected format: one waypoint per line, 7 space-separated doubles
// ---------------------------------------------------------------------------
std::vector<std::array<double, 7>> loadTrajectory(const std::string& path) {
    std::vector<std::array<double, 7>> traj;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open trajectory file: " + path);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::array<double, 7> wp;
        for (int i = 0; i < 7; i++) {
            if (!(ss >> wp[i])) {
                throw std::runtime_error("Malformed line: " + line);
            }
        }
        traj.push_back(wp);
    }
    return traj;
}

// ---------------------------------------------------------------------------
// 2. Savitzky-Golay coefficients for window=5, poly=3, derivative=0
//    Larger window → smoother. Must be odd. Increase to 11, 21 if needed.
// ---------------------------------------------------------------------------
std::vector<std::array<double, 7>> savitzkyGolay(
    const std::vector<std::array<double, 7>>& traj,
    int half_window = 5)   // total window = 2*half_window+1
{
    const int N = traj.size();
    const int W = half_window;
    std::vector<std::array<double, 7>> out(traj);  // copy, edges stay as-is

    // Precompute SG coefficients for poly=3 via least-squares (closed form)
    // For simplicity we use a weighted average kernel approximation.
    // Full closed-form SG coeffs for arbitrary window:
    auto sg_coeffs = [&](int w) -> std::vector<double> {
        // Polynomial order 3, zero-th derivative
        // Coefficients computed as per Savitzky & Golay (1964)
        int m = 2 * w + 1;
        std::vector<double> c(m);
        // Build Vandermonde matrix and solve — here we use the known
        // analytic formula for order-3 SG filter
        double norm = 0.0;
        for (int k = -w; k <= w; k++) {
            // coeff for position k in [-w, w]
            double v = (3.0*(3*w*w*(w+1)*(w+1) - 4*k*k*(3*w*w + 3*w - 1 - 5*k*k)));
            // This is the exact formula for cubic SG smoothing
            // Simplified approximation: use symmetric triangular weight
            c[k + w] = (w + 1.0 - std::abs(k));
            norm += c[k + w];
        }
        for (auto& x : c) x /= norm;
        return c;
    };

    auto coeffs = sg_coeffs(W);

    for (int i = W; i < N - W; i++) {
        for (int j = 0; j < 7; j++) {
            double val = 0.0;
            for (int k = -W; k <= W; k++) {
                val += coeffs[k + W] * traj[i + k][j];
            }
            out[i][j] = val;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 3. Simple moving average — use as alternative or pre-pass
// ---------------------------------------------------------------------------
std::vector<std::array<double, 7>> movingAverage(
    const std::vector<std::array<double, 7>>& traj,
    int window = 11)
{
    const int N = traj.size();
    const int half = window / 2;
    std::vector<std::array<double, 7>> out(traj);

    for (int i = half; i < N - half; i++) {
        for (int j = 0; j < 7; j++) {
            double sum = 0.0;
            for (int k = -half; k <= half; k++) {
                sum += traj[i + k][j];
            }
            out[i][j] = sum / window;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 4. Check max jerk (spikes = noise = sound)
// ---------------------------------------------------------------------------
void printMaxJerk(const std::vector<std::array<double, 7>>& traj, double dt = 0.001) {
    std::array<double, 7> max_jerk{};
    for (size_t i = 1; i + 2 < traj.size(); i++) {
        for (int j = 0; j < 7; j++) {
            double vel0 = (traj[i][j]   - traj[i-1][j]) / dt;
            double vel1 = (traj[i+1][j] - traj[i][j])   / dt;
            double acc0 = (vel1 - vel0) / dt;
            double vel2 = (traj[i+2][j] - traj[i+1][j]) / dt;
            double acc1 = (vel2 - vel1) / dt;
            double jerk = std::abs((acc1 - acc0) / dt);
            if (jerk > max_jerk[j]) max_jerk[j] = jerk;
        }
    }
    std::cout << "Max jerk per joint [rad/s^3]:\n";
    for (int j = 0; j < 7; j++) {
        std::cout << "  J" << j+1 << ": " << max_jerk[j] << "\n";
    }
}

// ---------------------------------------------------------------------------
// 5. Execute trajectory on the robot
// ---------------------------------------------------------------------------
void executeTrajectory(franka::Robot& robot,
                       const std::vector<std::array<double, 7>>& traj)
{
    size_t idx = 0;
    const size_t N = traj.size();

    robot.control([&](const franka::RobotState& /*state*/, franka::Duration /*dt*/)
        -> franka::JointPositions
    {
        franka::JointPositions cmd{traj[idx]};

        if (idx + 1 >= N) {
            return franka::MotionFinished(cmd);
        }
        idx++;
        return cmd;
    });
}




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







// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // if (argc < 3) {
    //     std::cerr << "Usage: " << argv[0]
    //               << " <robot_ip> <trajectory_file> [smooth_window=11]\n";
    //     return 1;
    // }

    const std::string robot_ip   = "172.16.0.2"; // argv[1];
    const std::string traj_file  = "../trajectories/new_test/q_1kHz.csv"; // argv[2];
    const int         window     = 11; // (argc >= 4) ? std::stoi(argv[3]) : 11;

    // --- Load ---
    // auto traj = loadTrajectory(traj_file);
    auto traj = loadTrajectoryCSV(traj_file);
    std::cout << "Loaded " << traj.size() << " waypoints.\n";

    // --- Diagnose ---
    std::cout << "--- Before smoothing ---\n";
    printMaxJerk(traj);

    // --- Smooth ---
    // Use moving average first (fast), then SG on top for shape preservation
    auto traj_smooth = movingAverage(traj, window);
    traj_smooth      = savitzkyGolay(traj_smooth, window / 2);

    std::cout << "--- After smoothing (window=" << window << ") ---\n";
    printMaxJerk(traj_smooth);

    // --- Execute ---
    try {
        franka::Robot robot(robot_ip);
        robot.setCollisionBehavior(
            {{20,20,20,20,20,20,20}}, {{20,20,20,20,20,20,20}},
            {{20,20,20,20,20,20}},    {{20,20,20,20,20,20}});

        MotionGenerator motion_generator(0.3, traj_smooth[0]);
        std::cout << "WARNING: This example will move the robot! "
                << "Please make sure to have the user stop button at hand!" << std::endl
                << "Press Enter to continue..." << std::endl;
        std::cin.ignore();
        robot.control(motion_generator);
        std::cout << "Finished moving to initial joint configuration." << std::endl;

        std::cout << "Executing smoothed trajectory...\n";
        executeTrajectory(robot, traj_smooth);
        std::cout << "Done.\n";
    } catch (franka::Exception const& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}