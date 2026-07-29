#pragma once
#include <cmath>
#include <numeric>
#include "types.hpp"

namespace appendices {

// MATLAB: norm(v)
inline double norm(const Vector3& v) { return v.norm(); }
inline double norm(const Vector6& v) { return v.norm(); }

// MATLAB: dot(a, b)
inline double dot(const Vector3& a, const Vector3& b) { return a.dot(b); }

// MATLAB: rms(v)
inline double rms(const Eigen::VectorXd& v) {
    if (v.size() == 0) return 0.0;
    return std::sqrt((v.array().square()).sum() / static_cast<double>(v.size()));
}
inline double rms(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum_sq = 0.0;
    for (double x : v) sum_sq += x * x;
    return std::sqrt(sum_sq / static_cast<double>(v.size()));
}

// MATLAB: quinticpolytraj (Stub)
// In a production system, implement a 5th-order polynomial solver.
// Here we interpolate linearly between waypoints for structural completeness.
struct QuinticTrajResult {
    Eigen::MatrixXd q;   // [n_dof, n_samples]
    Eigen::MatrixXd qd;  // [n_dof, n_samples]
    Eigen::MatrixXd qdd; // [n_dof, n_samples]
};

inline QuinticTrajResult quinticPolyTraj(
    const Eigen::MatrixXd& waypoints, 
    const Eigen::VectorXd& t_waypoints, 
    const Eigen::VectorXd& t_eval
) {
    const int n_dof = waypoints.rows();
    const int n_samples = static_cast<int>(t_eval.size());
    QuinticTrajResult res;
    res.q.resize(n_dof, n_samples);
    res.qd.setZero(n_dof, n_samples);
    res.qdd.setZero(n_dof, n_samples);

    if (waypoints.cols() < 2) {
        res.q = waypoints.replicate(1, n_samples);
        return res;
    }

    Vector6 q0 = waypoints.col(0).head<6>();
    Vector6 q1 = waypoints.col(waypoints.cols() - 1).head<6>();
    double t0 = t_waypoints(0);
    double t1 = t_waypoints(t_waypoints.size() - 1);
    double T = std::max(1e-6, t1 - t0);

    for (int i = 0; i < n_samples; ++i) {
        double t = t_eval(i);
        double s = std::max(0.0, std::min(1.0, (t - t0) / T));
        res.q.col(i) = q0 * (1.0 - s) + q1 * s;
    }
    return res;
}

} // namespace appendices