#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

namespace collision_avoidance {

// Skeleton segment indices (pairs of joint indices)
extern const std::array<std::array<int, 2>, 10> skel_index;

// Robot capsule radii (rv)
extern const std::array<double, 4> rv;

// Safety parameters
extern const double T_reaction;   // reaction time [s]
extern const double csi;          // safety constant
extern const double max_human_speed;
extern const double max_robot_speed;

// Robot DH parameters (7 rows, 4 columns: a, alpha, d, theta_offset)
// Matches your Python DH matrix
extern const Eigen::Matrix<double, 7, 4> DH_default;

// Robot to marker offset
extern const Eigen::Vector3d robot_to_marker_pos;

// Human skeleton segment safety radii (r_sw_h)
extern const std::array<double, 10> r_sw_h;

// Utility
double clamp(double n);
double trapz(const Eigen::VectorXd& t, const std::vector<double>& y);

// 5th-order polynomial trajectory
// Returns [q, q_dot, q_ddot] as matrices (3 x N)
Eigen::Matrix<double, 3, Eigen::Dynamic> TrajPoly5(
    double qi, double qi_p, double qi_pp,
    double qf, double qf_p, double qf_pp,
    double traj_duration, double time_step);

// Forward kinematics: compute capsule end point in robot base frame
// DH: 7x4 matrix, q: 7-vector, k: number of links (1..7)
Eigen::Vector3d compute_capsule_end_point(
    const Eigen::Matrix<double, 7, 4>& DH,
    const Eigen::Vector<double, 7>& q,
    int k);

// Jacobian for a point A_k given joint positions/velocities and DH
// q, dq: up to 7 joints
// DH: subset rows (n_joints x 4)
// A_k: 3D point in base frame
// Returns 6 x n_joints matrix [J_linear; J_angular]
Eigen::Matrix<double, 6, Eigen::Dynamic> Jacobian(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::MatrixXd& DH,
    const Eigen::Vector3d& A_k);

// Compute max velocity projection for 4 robot capsules
// q, q_p: 7-vectors
// DH: 7x4
// rv: 4 capsule radii
// C_h, C_r: closest points on human segment and robot capsule axis
// Returns 4 velocities (one per capsule)
std::array<double, 4> compute_max_vel_capsule(
    const Eigen::Vector<double, 7>& q,
    const Eigen::Vector<double, 7>& q_p,
    const Eigen::Matrix<double, 7, 4>& DH,
    const std::array<double, 4>& rv,
    const Eigen::Vector3d& C_h,
    const Eigen::Vector3d& C_r);

// Compute robot capsules (4 segments, each with A and B endpoints)
// q: 7-vector
// DH: 7x4
// robot_to_marker_pos: offset to subtract
// Returns 4x2x3 array: [capsule_id][endpoint_id(0=A,1=B)][xyz]
std::array<std::array<Eigen::Vector3d, 2>, 4> compute_robot_capsules(
    const Eigen::Vector<double, 7>& q,
    const Eigen::Matrix<double, 7, 4>& DH,
    const Eigen::Vector3d& robot_to_marker_pos);

// Distance between two segments:
// P1-Q1: human segment
// P2-Q2: robot capsule segment
// Returns [distance, C_h, C_r]
struct SegmentDistanceResult {
    double distance;
    Eigen::Vector3d C_h;
    Eigen::Vector3d C_r;
};

SegmentDistanceResult distance_to_segment(
    const Eigen::Vector3d& P1,
    const Eigen::Vector3d& Q1,
    const Eigen::Vector3d& P2,
    const Eigen::Vector3d& Q2);

// Distance from robot capsule to skeleton
// skeleton: vector of 15 joints, each a 3D point (if invalid, use NaNs)
// P2, Q2: robot capsule endpoints
// Returns {distance, C_h, C_r, index_of_segment} or nullopt if no valid segment
struct SkeletonDistanceResult {
    double distance;
    Eigen::Vector3d C_h;
    Eigen::Vector3d C_r;
    int ind_h; // index in skel_index
};

std::optional<SkeletonDistanceResult> distance_to_skeleton(
    const std::array<Eigen::Vector3d, 15>& skeleton,
    const Eigen::Vector3d& P2,
    const Eigen::Vector3d& Q2);

// Compute safety radii for robot capsules (r_sw_r)
// T_stop: stopping time [s]
// q, q_p, q_pp: current joint pos/vel/acc (7-vectors)
// C_h, C_r: closest points between human and robot capsule axes
// DH: 7x4
// rv: 4 capsule radii
// Returns 4 safety radii
std::array<double, 4> capsule_calculation(
    double T_stop,
    const Eigen::Vector<double, 7>& q,
    const Eigen::Vector<double, 7>& q_p,
    const Eigen::Vector<double, 7>& q_pp,
    const Eigen::Vector3d& C_h,
    const Eigen::Vector3d& C_r,
    const Eigen::Matrix<double, 7, 4>& DH,
    const std::array<double, 4>& rv);

// Main collision check:
// skeleton: 15 joints (3D each)
// q, q_p, q_pp: robot state (7-vectors)
// T_stop: stopping time
// DH: 7x4
// Returns true if stop flag should be raised
bool FlagStopServer(
    const std::array<Eigen::Vector3d, 15>& skeleton,
    const Eigen::Vector<double, 7>& q,
    const Eigen::Vector<double, 7>& q_p,
    const Eigen::Vector<double, 7>& q_pp,
    double T_stop,
    const Eigen::Matrix<double, 7, 4>& DH,
    const std::array<double, 4>& rv,
    const std::array<double, 10>& r_sw_h,
    const Eigen::Vector3d& robot_to_marker_pos);

} // namespace collision_avoidance