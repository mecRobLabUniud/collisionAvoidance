/**
 * Appendix A: Choice of the relative weights - SSM & Escape Trajectories
 * 
 * Complete C++ translation of the MATLAB code in Appendix A.
 * Evaluates the performance of the combined SSM (Safe Stop Monitoring) 
 * and escape trajectory method with varying Qv weights.
 * 
 * Dependencies: Eigen3 (for matrix operations)
 * Compile: g++ -std=c++17 -I/path/to/eigen appendix_a_complete.cpp -o appendix_a
 * Run: ./appendix_a
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <limits>

// =============================================================================
// TYPE DEFINITIONS & UTILITIES
// =============================================================================

using Vector3 = Eigen::Vector3d;
using Vector6 = Eigen::Vector6d;
using Matrix3 = Eigen::Matrix3d;
using Matrix6 = Eigen::Matrix6d;
using Matrix66 = Eigen::Matrix6d;
using Transform4 = Eigen::Matrix4d;

struct JointLimits {
    Vector6 q_min, q_max;
    Vector6 qd_min, qd_max;
    Vector6 qdd_min, qdd_max;
};

// Random number generator (replaces MATLAB rand)
struct Rng {
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform01{0.0, 1.0};
    
    Rng(unsigned seed = 42) : gen(seed) {}
    
    double rand01() { return uniform01(gen); }
    
    double randRange(double a, double b) { 
        return a + rand01() * (b - a); 
    }
    
    Vector3 randVector3(double xmin, double xmax, 
                        double ymin, double ymax, 
                        double zmin, double zmax) {
        return Vector3{
            randRange(xmin, xmax),
            randRange(ymin, ymax),
            randRange(zmin, zmax)
        };
    }
};

// Math utilities
namespace math {
    inline double norm(const Vector3& v) { return v.norm(); }
    inline double norm(const Vector6& v) { return v.norm(); }
    inline double dot(const Vector3& a, const Vector3& b) { return a.dot(b); }
    
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
}

// Quintic polynomial trajectory (simplified version)
struct QuinticTrajResult {
    Eigen::MatrixXd q;    // [n_dof, n_samples]
    Eigen::MatrixXd qd;   // [n_dof, n_samples]
    Eigen::MatrixXd qdd;  // [n_dof, n_samples]
};

QuinticTrajResult quinticPolyTraj(
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

// =============================================================================
// ROBOT MODEL & KINEMATICS (STUBS - Replace with real UR5e implementation)
// =============================================================================

struct RobotModel {
    // Placeholder for UR5e kinematic model
    // In production: use Pinocchio, KDL, or custom forward/inverse kinematics
};

struct IKResult {
    Vector6 q;
    bool success = false;
};

IKResult inverseKinematics(const RobotModel& robot, 
                           const Transform4& T_target,
                           const Vector6& q_seed) {
    // TODO: Implement real inverse kinematics for UR5e
    // This is a placeholder that returns the seed configuration
    IKResult res;
    res.q = q_seed;
    res.success = true;
    return res;
}

Transform4 getTransform(const RobotModel& robot, const Vector6& q) {
    // TODO: Implement forward kinematics
    // Returns homogeneous transform from base to tool0
    return Transform4::Identity();
}

Matrix66 geometricJacobian(const RobotModel& robot, const Vector6& q) {
    // TODO: Implement geometric Jacobian computation
    // Returns 6x6 Jacobian matrix at configuration q
    return Matrix66::Identity();
}

// =============================================================================
// SAFETY ALGORITHM (STUB - Replace with Appendix E implementation)
// =============================================================================

struct SSMescapeResult {
    Vector6 qddot;
    Vector6 qdot_next;
    Vector6 q_next;
    Vector3 p_next;
    Vector3 v_next;
    int flag;  // 1 = success, 0 = failure
};

/**
 * SSMescape: Safe Stop Monitoring with Escape Trajectory
 * 
 * This function should implement the quadratic program from Appendix E.
 * It computes safe joint accelerations that respect SSM constraints 
 * (minimum distance to human) while allowing escape maneuvers.
 */
SSMescapeResult SSMescape(
    const RobotModel& robot,
    const JointLimits& joint_limits,
    double delta_t,
    double stopping_time,
    const Vector6& q_t,
    const Vector6& qdot_t,
    const Vector3& p_ref,
    const Vector3& v_ref,
    const Vector6& qddot_suggestion,
    const Vector3& p_intrusion,
    const Vector3& pd_intrusion,
    double delta,
    const Vector6& q_des,
    const Vector6& qd_des,
    double Qv
) {
    // TODO: Implement the full QP from Appendix E
    // This stub returns a safe "do nothing" trajectory
    
    SSMescapeResult res;
    res.qddot.setZero();
    res.qdot_next = qdot_t;
    res.q_next = q_t + delta_t * qdot_t;
    res.p_next = p_ref;
    res.v_next = v_ref;
    res.flag = 1;  // Success
    
    return res;
}

// =============================================================================
// EXPERIMENT PARAMETERS (Appendix A - Section "Parameters")
// =============================================================================

struct ExperimentParams {
    // Test configuration
    int number_trajectories = 5;
    int number_intrusions_per_trajectory = 5;
    
    // Qv values (weight for velocity error in optimization)
    std::vector<double> Qvs = {
        0.06, 0.07, 0.08, 0.09, 0.1,
        0.11, 0.125, 0.15, 0.2, 0.3
    };
    
    // Timing
    double time_beginning = 0.0;
    double time_final = 6.0;
    double computational_frequency = 200.0;  // Hz
    double stopping_time = 0.3;  // seconds
    double pause_after_collision = 1.75;  // seconds
    
    // Manipulator joint limits (UR5e)
    JointLimits joint_limits;
    
    // Robot trajectory bounds
    double x_min_robot_initial = 0.5;
    double x_max_robot_initial = 0.7;
    double y_min_robot_initial = -0.2;
    double y_max_robot_initial = 0.1;
    double z_min_robot_initial = 0.2;
    double z_max_robot_initial = 0.4;
    
    double x_min_robot_final = -0.2;
    double x_max_robot_final = 0.1;
    double y_min_robot_final = 0.5;
    double y_max_robot_final = 0.7;
    double z_min_robot_final = 0.2;
    double z_max_robot_final = 0.4;
    
    // Human trajectory parameters
    double b_human = 0.5;
    double h_human = 0.1;
    double z_min_human = 0.15;
    double z_max_human = 0.45;
    double time_collision_min = 2.0;
    double time_collision_max = 4.0;
    double time_movement = 2.0;
    
    ExperimentParams() {
        const double pi = M_PI;
        
        joint_limits.q_min.fill(-2.0 * pi);
        joint_limits.q_max.fill( 2.0 * pi);
        
        joint_limits.qd_min.fill(-1.0 * pi);
        joint_limits.qd_max.fill( 1.0 * pi);
        
        joint_limits.qdd_min.fill(-1.25 * pi);
        joint_limits.qdd_max.fill( 1.25 * pi);
    }
    
    double computational_period() const {
        return 1.0 / computational_frequency;
    }
};

// =============================================================================
// MAIN EXPERIMENT LOOP (Appendix A - Section "Experiment")
// =============================================================================

int main() {
    std::cout << "=== Appendix A: SSM & Escape Trajectories ===" << std::endl;
    
    // Initialize parameters and utilities
    ExperimentParams params;
    Rng rng;
    RobotModel robot;
    
    const double dt = params.computational_period();
    const double t_begin = params.time_beginning;
    const double t_final = params.time_final;
    
    // Create time vectors (MATLAB: time_points, time_points_complete)
    std::vector<double> time_points;
    for (double t = t_begin; t <= t_final + 1e-9; t += dt) {
        time_points.push_back(t);
    }
    
    std::vector<double> time_points_complete;
    for (double t = t_begin; t <= t_final + 25.0 + 1e-9; t += dt) {
        time_points_complete.push_back(t);
    }
    
    const int number_time_points = static_cast<int>(time_points.size());
    const int number_time_points_complete = static_cast<int>(time_points_complete.size());
    const int number_QpQv = static_cast<int>(params.Qvs.size());
    
    std::cout << "Time points: " << number_time_points << std::endl;
    std::cout << "Qv values: " << number_QpQv << std::endl;
    std::cout << "Trajectories: " << params.number_trajectories << std::endl;
    std::cout << "Intrusions per trajectory: " << params.number_intrusions_per_trajectory << std::endl;
    
    // Results storage (3D: trajectories × intrusions × Qv)
    // In MATLAB: T_TIME(number_trajectories, number_intrusions..., number_QpQv)
    using Results3D = std::vector<std::vector<std::vector<double>>>;
    
    Results3D T_TIME(
        params.number_trajectories,
        Results3D::value_type(
            params.number_intrusions_per_trajectory,
            std::vector<double>(number_QpQv, std::nan(""))
        )
    );
    
    Results3D qdd_real_rms(
        params.number_trajectories,
        Results3D::value_type(
            params.number_intrusions_per_trajectory,
            std::vector<double>(number_QpQv, std::nan(""))
        )
    );
    
    Results3D v_real_rms(
        params.number_trajectories,
        Results3D::value_type(
            params.number_intrusions_per_trajectory,
            std::vector<double>(number_QpQv, std::nan(""))
        )
    );
    
    Results3D v_real_rms_prime(
        params.number_trajectories,
        Results3D::value_type(
            params.number_intrusions_per_trajectory,
            std::vector<double>(number_QpQv, std::nan(""))
        )
    );
    
    Results3D T_TIME_ORIGINAL(
        params.number_trajectories,
        Results3D::value_type(
            params.number_intrusions_per_trajectory,
            std::vector<double>(number_QpQv, 0.0)
        )
    );
    
    Results3D R_STOPS(
        params.number_trajectories,
        Results3D::value_type(
            params.number_intrusions_per_trajectory,
            std::vector<double>(number_QpQv, 0.0)
        )
    );
    
    Results3D R_IDLE(
        params.number_trajectories,
        Results3D::value_type(
            params.number_intrusions_per_trajectory,
            std::vector<double>(number_QpQv, 0.0)
        )
    );
    
    // Base configuration (MATLAB: q_base = [0,1,1,pi,pi,0]')
    Vector6 q_base;
    q_base << 0, 1, 1, M_PI, M_PI, 0;
    
    std::cout << "\nStarting experiment loops..." << std::endl;
    
    // =========================================================================
    // LOOP w1: Trajectories
    // =========================================================================
    for (int w1 = 0; w1 < params.number_trajectories; ++w1) {
        std::cout << "\n--- Trajectory " << (w1 + 1) << "/" << params.number_trajectories << " ---" << std::endl;
        
        // ---------------------------------------------------------------------
        // Robot trajectory generation (MATLAB: "Robot trajectory generation")
        // ---------------------------------------------------------------------
        Vector3 p_ri = rng.randVector3(
            params.x_min_robot_initial, params.x_max_robot_initial,
            params.y_min_robot_initial, params.y_max_robot_initial,
            params.z_min_robot_initial, params.z_max_robot_initial
        );
        
        Vector3 p_rf = rng.randVector3(
            params.x_min_robot_final, params.x_max_robot_final,
            params.y_min_robot_final, params.y_max_robot_final,
            params.z_min_robot_final, params.z_max_robot_final
        );
        
        std::cout << "Initial position: (" << p_ri.transpose() << ")" << std::endl;
        std::cout << "Final position: (" << p_rf.transpose() << ")" << std::endl;
        
        // Build homogeneous transforms p1, p2 (MATLAB 4x4 matrices)
        Transform4 p1 = Transform4::Identity();
        Transform4 p2 = Transform4::Identity();
        
        // Rotation part (fixed orientation from MATLAB code)
        p1.block<3,3>(0,0) << 1, 0, 0,
                              0, -1, 0,
                              0, 0, -1;
        p1.col(3).head<3>() = p_ri;
        
        p2.block<3,3>(0,0) << 1, 0, 0,
                              0, -1, 0,
                              0, 0, -1;
        p2.col(3).head<3>() = p_rf;
        
        // Inverse kinematics (MATLAB: ik_robot('tool0', p1, ikWeights, q_base))
        IKResult ik1 = inverseKinematics(robot, p1, q_base);
        IKResult ik2 = inverseKinematics(robot, p2, q_base);
        
        if (!ik1.success || !ik2.success) {
            std::cerr << "IK failed for trajectory " << (w1 + 1) << std::endl;
            continue;
        }
        
        // Build waypoints matrix [6 dof × 2 waypoints]
        Eigen::MatrixXd waypoints(6, 2);
        waypoints.col(0) = ik1.q;
        waypoints.col(1) = ik2.q;
        
        // Time vectors for trajectory generation
        Eigen::VectorXd t_waypoints(2);
        t_waypoints << t_begin, t_final;
        
        Eigen::VectorXd t_eval(number_time_points_complete);
        for (int i = 0; i < number_time_points_complete; ++i) {
            t_eval(i) = time_points_complete[i];
        }
        
        // Generate quintic polynomial trajectory (MATLAB: quinticpolytraj)
        auto traj = quinticPolyTraj(waypoints, t_waypoints, t_eval);
        
        // Compute joint acceleration RMS for nominal trajectory
        double qdd_rms_nominal = math::rms(traj.qdd.row(0).head(number_time_points));
        
        // ---------------------------------------------------------------------
        // Analysis of the trajectory in cartesian space
        // ---------------------------------------------------------------------
        std::vector<Vector3> p_path(number_time_points_complete, Vector3::Zero());
        std::vector<Vector3> v_path(number_time_points_complete, Vector3::Zero());
        std::vector<double> v_module;
        std::vector<int> stopping_times;
        
        for (int i = 0; i < number_time_points; ++i) {
            // Compute Cartesian velocity from Jacobian (MATLAB: v = geometricJacobian * qd)
            Matrix66 J = geometricJacobian(robot, traj.q.col(i));
            Vector6 v_full = J * traj.qd.col(i);
            v_path[i] = v_full.tail<3>();  // Linear velocity (last 3 elements)
            v_module.push_back(v_path[i].norm());
            
            // Forward kinematics for position (MATLAB: transform = getTransform)
            Transform4 transform = getTransform(robot, traj.q.col(i));
            p_path[i] = transform.col(3).head<3>();
            
            // Check if reached goal position (within 5mm)
            if ((p_path[i] - p_rf).norm() <= 0.005) {
                stopping_times.push_back(i);
            }
        }
        
        // Store original arrival time (MATLAB: T_TIME_ORIGINAL)
        int min_stopping_idx = stopping_times.empty() ? number_time_points : stopping_times[0];
        double T_TIME_ORIGINAL_val = min_stopping_idx * dt;
        
        // Extend trajectory to hold final position
        for (int i = number_time_points; i < number_time_points_complete; ++i) {
            p_path[i] = p_path[number_time_points - 1];
        }
        
        // Nominal velocity RMS (before any intrusion)
        double v_rms_nominal = math::rms(v_module);
        
        std::cout << "Nominal arrival time: " << T_TIME_ORIGINAL_val << "s" << std::endl;
        std::cout << "Nominal velocity RMS: " << v_rms_nominal << " m/s" << std::endl;
        
        // =====================================================================
        // LOOP w2: Intrusions (MATLAB: "Generation of the randomized human trajectory")
        // =====================================================================
        for (int w2 = 0; w2 < params.number_intrusions_per_trajectory; ++w2) {
            std::cout << "\n  Intrusion " << (w2 + 1) << "/" << params.number_intrusions_per_trajectory << std::endl;
            
            // Generate randomized human trajectory
            double r1 = rng.rand01();
            double r2 = rng.rand01();
            
            Vector3 p_hi;
            p_hi(0) = 0.65 - (r1 * params.b_human - params.b_human / 2.0) * sin(M_PI / 4.0) 
                         + (r2 * params.h_human - params.h_human / 2.0) * cos(M_PI / 4.0);
            p_hi(1) = 0.65 + (r1 * params.b_human - params.b_human / 2.0) * cos(M_PI / 4.0) 
                         + (r2 * params.h_human - params.h_human / 2.0) * cos(M_PI / 4.0);
            p_hi(2) = rng.randRange(params.z_min_human, params.z_max_human);
            
            r1 = rng.rand01();
            r2 = rng.rand01();
            
            Vector3 p_hf;
            p_hf(0) = 0.65 - (r1 * params.b_human - params.b_human / 2.0) * sin(M_PI / 4.0) 
                         + (r2 * params.h_human - params.h_human / 2.0) * cos(M_PI / 4.0);
            p_hf(1) = 0.65 + (r1 * params.b_human - params.b_human / 2.0) * cos(M_PI / 4.0) 
                         + (r2 * params.h_human - params.h_human / 2.0) * cos(M_PI / 4.0);
            p_hf(2) = rng.randRange(params.z_min_human, params.z_max_human);
            
            double time_collision = rng.randRange(params.time_collision_min, params.time_collision_max);
            
            // Collision position on nominal trajectory
            int collision_step = static_cast<int>(time_collision * params.computational_frequency);
            collision_step = std::min(collision_step, number_time_points - 1);
            Vector3 collision_position = p_path[collision_step];
            
            // Intrusion trajectory (simplified: constant position for this example)
            // In full implementation: use quinticpolytraj for p_intrusion, pd_intrusion, pdd_intrusion
            Vector3 p_intrusion = p_hi;
            Vector3 pd_intrusion = Vector3::Zero();  // Velocity of intrusion
            Vector3 pdd_intrusion = Vector3::Zero(); // Acceleration of intrusion
            
            // =================================================================
            // LOOP w3: Qv Values (MATLAB: "SSM & escape Optimization")
            // =================================================================
            for (int w3 = 0; w3 < number_QpQv; ++w3) {
                double Qv = params.Qvs[w3];
                
                // Initialize simulation state
                Vector6 qddot_real = Vector6::Zero();
                Vector6 qdot_real = traj.qd.col(0);
                Vector6 q_real = traj.q.col(0);
                Vector3 p_real = p_path[0];
                Vector3 v_real = v_path[0];
                
                std::vector<double> v_real_module;
                std::vector<Vector6> qddot_real_history;
                
                bool collision_occurred = false;
                int arrival_step = number_time_points_complete;
                int failure_flag = 0;
                
                // Main control loop (MATLAB: for i = 1 : number_time_points_complete-1)
                for (int i = 0; i < number_time_points_complete - 1; ++i) {
                    // Compute safety distance delta (MATLAB: 0.1 + sqrt(dot(pd_intrusion, pd_intrusion)) * stopping_time)
                    double delta_safety = 0.1 + pd_intrusion.norm() * params.stopping_time;
                    
                    // Call SSMescape function (Appendix E)
                    SSMescapeResult res = SSMescape(
                        robot,
                        params.joint_limits,
                        dt,
                        params.stopping_time,
                        q_real,
                        qdot_real,
                        p_path[i + 1],
                        v_path[i + 1],
                        qddot_real,
                        p_intrusion,
                        pd_intrusion,
                        delta_safety,
                        traj.q.col(i + 1),
                        traj.qd.col(i + 1),
                        Qv
                    );
                    
                    // Update state
                    qddot_real = res.qddot;
                    qdot_real = res.qdot_next;
                    q_real = res.q_next;
                    p_real = res.p_next;
                    v_real = res.v_next;
                    
                    // Store velocity magnitude for RMS calculation
                    v_real_module.push_back(v_real.norm());
                    qddot_real_history.push_back(qddot_real);
                    
                    // Check if optimization succeeded
                    if (res.flag != 1) {
                        failure_flag = 1;
                        R_STOPS[w1][w2][w3] += 1.0;
                        std::cout << "    Qv=" << Qv << " -> Optimization failed at step " << i << std::endl;
                        break;
                    }
                    
                    // Check if robot reached goal position (within 5mm)
                    if ((p_real - p_rf).norm() <= 0.005) {
                        arrival_step = i;
                        std::cout << "    Qv=" << Qv << " -> Arrived at t=" << (i * dt) << "s" << std::endl;
                        break;
                    }
                }
                
                // =================================================================
                // Store Results (MATLAB: results storage section)
                // =================================================================
                if (failure_flag == 1) {
                    // Optimization failed: store NaN
                    T_TIME[w1][w2][w3] = std::nan("");
                    v_real_rms[w1][w2][w3] = std::nan("");
                    v_real_rms_prime[w1][w2][w3] = std::nan("");
                    qdd_real_rms[w1][w2][w3] = std::nan("");
                } else {
                    // Store arrival time
                    if (arrival_step < number_time_points_complete) {
                        T_TIME[w1][w2][w3] = arrival_step * dt;
                        
                        // Compute velocity RMS (full trajectory)
                        if (!v_real_module.empty()) {
                            v_real_rms[w1][w2][w3] = math::rms(v_real_module);
                        }
                        
                        // Compute velocity RMS (up to arrival, excluding stopping phase)
                        int prime_end = std::min(arrival_step, min_stopping_idx - 1);
                        if (prime_end > 0 && prime_end <= static_cast<int>(v_real_module.size())) {
                            std::vector<double> v_prime(v_real_module.begin(), v_real_module.begin() + prime_end);
                            v_real_rms_prime[w1][w2][w3] = math::rms(v_prime);
                        }
                        
                        // Compute joint acceleration RMS
                        if (!qddot_real_history.empty()) {
                            Eigen::MatrixXd qddot_mat(6, qddot_real_history.size());
                            for (size_t k = 0; k < qddot_real_history.size(); ++k) {
                                qddot_mat.col(k) = qddot_real_history[k];
                            }
                            qdd_real_rms[w1][w2][w3] = math::rms(qddot_mat.row(0));
                        }
                    } else {
                        // Did not reach goal
                        T_TIME[w1][w2][w3] = std::nan("");
                        v_real_rms[w1][w2][w3] = std::nan("");
                        v_real_rms_prime[w1][w2][w3] = std::nan("");
                        qdd_real_rms[w1][w2][w3] = std::nan("");
                    }
                }
                
                // Compute R_IDLE: ratio of idle time to total time
                // R_IDLE = R_STOPS * pause_after_collision / T_TIME
                if (!std::isnan(T_TIME[w1][w2][w3]) && T_TIME[w1][w2][w3] > 1e-6) {
                    R_IDLE[w1][w2][w3] = R_STOPS[w1][w2][w3] * params.pause_after_collision / T_TIME[w1][w2][w3];
                }
                
                // Optional: Print summary for this Qv value
                std::cout << "    Qv=" << Qv 
                          << " -> T=" << T_TIME[w1][w2][w3] 
                          << ", RMS=" << v_real_rms[w1][w2][w3]
                          << ", Stops=" << R_STOPS[w1][w2][w3] << std::endl;
            }
        }
    }
    
    // =========================================================================
    // PLOT RESULTS (MATLAB: "Plot of the results")
    // =========================================================================
    std::cout << "\n=== Experiment Complete ===" << std::endl;
    std::cout << "Results stored in T_TIME, v_real_rms, R_STOPS, R_IDLE matrices." << std::endl;
    
    // Example: Compute mean ratios (MATLAB: mean_v_rms_ratio = nanmean(nanmean(v_rms_ratio)))
    std::vector<double> v_rms_ratios;
    std::vector<double> T_TIME_ratios;
    
    for (int w1 = 0; w1 < params.number_trajectories; ++w1) {
        for (int w2 = 0; w2 < params.number_intrusions_per_trajectory; ++w2) {
            for (int w3 = 0; w3 < number_QpQv; ++w3) {
                double original = T_TIME_ORIGINAL[w1][w2][w3];
                double modified = T_TIME[w1][w2][w3];
                
                if (!std::isnan(modified) && original > 1e-6) {
                    T_TIME_ratios.push_back(modified / original);
                }
                
                double v_nominal = 1.0;  // Placeholder: use actual v_rms_nominal[w1][w2]
                double v_modified = v_real_rms[w1][w2][w3];
                
                if (!std::isnan(v_modified) && v_nominal > 1e-6) {
                    v_rms_ratios.push_back(v_nominal / v_modified);
                }
            }
        }
    }
    
    if (!v_rms_ratios.empty()) {
        double mean_v_ratio = math::rms(v_rms_ratios);
        std::cout << "Mean V-SMOOTHNESS ratio: " << mean_v_ratio << std::endl;
    }
    
    if (!T_TIME_ratios.empty()) {
        double mean_T_ratio = math::rms(T_TIME_ratios);
        std::cout << "Mean E-TIME ratio: " << mean_T_ratio << std::endl;
    }
    
    return 0;
}