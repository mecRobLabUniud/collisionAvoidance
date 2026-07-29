#pragma once


#include <Eigen/Dense>
#include <vector>
#include <array>
#include <string>


// ── CSV I/O ────────────────────────────────────────────────────────────────
std::vector<std::array<double, 7>> load_trajectory_CSV(const std::string& path);


void save_trajectory_CSV(const std::string& path,
                         const std::vector<std::array<double, 7>>& traj);


std::vector<double> load_timestamps_CSV(const std::string& path);


void save_timestamps_CSV(const std::string& path,
                         const std::vector<double>& times);


// ── Binary I/O ────────────────────────────────────────────────────────────
std::vector<std::array<double, 7>> load_bin(const std::string& path);


void save_bin(const std::string& path,
              const std::vector<std::array<double, 7>>& traj);


// ── Finite-difference derivative estimation ─────────────────────────────────
Eigen::VectorXd estimate_velocities(const Eigen::VectorXd& t,
                                    const Eigen::VectorXd& q);


Eigen::VectorXd estimate_accelerations(const Eigen::VectorXd& t,
                                       const Eigen::VectorXd& q);


// ── Quintic Hermite basis function ──────────────────────────────────────────
// Given normalized parameter s ∈ [0,1] and segment length h,
// returns the quintic interpolated value.
//
//   p(s) = h00*p0 + h10*h*v0 + h20*h²*a0
//         + h01*p1 + h11*h*v1 + h21*h²*a1
//
// Declared inline + defined here in the header since it's a small,
// frequently-called function — this lets the compiler inline it at each
// call site instead of requiring a separate translation unit each time.
inline double quintic_hermite(double s, double h,
                              double p0, double v0, double a0,
                              double p1, double v1, double a1) {
    double s2 = s * s;
    double s3 = s2 * s;
    double s4 = s3 * s;
    double s5 = s4 * s;


    double h00 =  1.0 - 10*s3 + 15*s4 -  6*s5;
    double h10 =  s   -  6*s3 +  8*s4 -  3*s5;
    double h20 =  0.5*s2 - 1.5*s3 + 1.5*s4 - 0.5*s5;
    double h01 =  10*s3  - 15*s4  +  6*s5;
    double h11 = -4*s3   +  7*s4  -  3*s5;
    double h21 =  0.5*s3 -    s4  +  0.5*s5;


    return h00*p0 + h10*h*v0 + h20*h*h*a0
         + h01*p1 + h11*h*v1 + h21*h*h*a1;
}


// ── Per-joint quintic spline interpolation ──────────────────────────────────
Eigen::VectorXd quintic_spline_interp(const Eigen::VectorXd& t_low,
                                      const Eigen::VectorXd& q_low,
                                      const Eigen::VectorXd& t_high);


// ── Main interpolation entry point (100Hz -> 1kHz upsampling) ──────────────
std::vector<std::array<double, 7>> interpolate_to_1kHz(
        const std::vector<std::array<double, 7>>& traj_low,
        std::vector<double> time_low);


// ── Sanity checks ────────────────────────────────────────────────────────
void validate_trajectory(const std::vector<std::array<double, 7>>& traj,
                         double rate_hz = 1000.0);