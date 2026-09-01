#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <optional>
#include <charconv>
#include <cmath>

#include "trajectory_utils.hpp"
#include "data_transmitter.hpp"
#include "utils.hpp"
#include "min_distance_calculation.hpp"
#include "robot_model.hpp"

#include "COLLcheck.hpp"
// #include "SSMPFL.hpp"

// Global flag, set by the signal handler
std::atomic<bool> running{true};

void signal_handler(int signum) {
    (void)signum;
    running = false;
}


struct JointLimits {
    Eigen::VectorXd q_min, q_max;
    Eigen::VectorXd qd_min, qd_max;
    Eigen::VectorXd qdd_min, qdd_max;
};

struct Rng {
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform01{0.0, 1.0};
    
    Rng(unsigned seed = 42) : gen(seed) {}
    
    double rand01() { return uniform01(gen); }
    
    double randRange(double a, double b) { 
        return a + rand01() * (b - a); 
    }
    
    Eigen::Vector3d randVector3(double xmin, double xmax, 
                        double ymin, double ymax, 
                        double zmin, double zmax) {
        return Eigen::Vector3d{
            randRange(xmin, xmax),
            randRange(ymin, ymax),
            randRange(zmin, zmax)
        };
    }
};


struct ExperimentParams {
    // Test configuration
    int number_trajectories = 1;
    int number_intrusions_per_trajectory = 1;
    
    // Qv values (weight for velocity error in optimization)
    std::vector<double> Qvs = {
        0.06, 0.07, 0.08, 0.09, 0.1,
        0.11, 0.125, 0.15, 0.2, 0.3
    };
    
    // PFL velocity values (velocity scaling factor)
    std::vector<double> vel_PFL = {0.1, 0.2, 0.3, 0.4, 0.5};
    
    // Timing
    double time_beginning = 0.0;
    double time_final = 10.0;
    double computational_frequency = 16.0;  // Hz
    double stopping_time = 0.3;  // seconds
    double pause_after_collision = 2.0;  // seconds
    
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


double rms(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    
    double sum_squares = 0.0;
    for (double v : values) {
        sum_squares += v * v;
    }
    
    return std::sqrt(sum_squares / values.size());
}




// ─────────────────────────────────────────────────────────────────────────────
// SSM + PFL + Escape Trajectories strategy
// ─────────────────────────────────────────────────────────────────────────────
int SSM_PFL_escape(RobotModel& robot, 
        const Eigen::VectorXd q_r, 
        const Eigen::VectorXd qd_r, 
        const Eigen::VectorXd qdd_r,
        const Eigen::VectorXd p_h, 
        const Eigen::VectorXd pd_h, 
        const Eigen::VectorXd pdd_h) {
    std::cout << "=== Appendix B: PFL & SSM & Escape Trajectories ===" << std::endl;
    
    // Initialize parameters and utilities
    ExperimentParams params;
    Rng rng;
    
    const double dt = params.computational_period();
    const double t_begin = params.time_beginning;
    const double t_final = params.time_final;
    
    // Create time vectors
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
    const int numberPFL = static_cast<int>(params.vel_PFL.size());
    
    std::cout << "Time points: " << number_time_points << std::endl;
    std::cout << "Qv values: " << number_QpQv << std::endl;
    std::cout << "PFL velocities: " << numberPFL << std::endl;
    std::cout << "Trajectories: " << params.number_trajectories << std::endl;
    std::cout << "Intrusions per trajectory: " << params.number_intrusions_per_trajectory << std::endl;

    
    // Base configuration (MATLAB: q_base = [0,1,1,pi,pi,0]') //0.000000, -0.785398, 0.000000, -2.356194, 0.000000, 1.570796, 0.785398
    // Eigen::VectorXd q_base{0.000000, -0.785398, 0.000000, -2.356194, 0.000000, 1.570796, 0.785398};
    
    std::cout << "\nStarting experiment loops..." << std::endl;
    
    
    // =========================================================================
    // LOOP w1: Trajectories
    // =========================================================================
    for (int w1 = 0; w1 < params.number_trajectories; ++w1) {
        

        std::cout << "HERE" << std::endl;

        Eigen::MatrixXd J = robot.ComputeJacobian("panda_link8", q_r);
        std::cout << "Jacobian (6x7):\n" << J << std::endl;

        Eigen::VectorXd v = J * qd_r;
        Eigen::Vector3d pd_r = v.tail<3>(); 
        Eigen::Vector3d p_r = robot.GetJointPose("panda_link8", q_r).translation().transpose();

        std::cout << "EE position: " << p_r << std::endl;
        std::cout << "EE velocity: " << pd_r << std::endl;


        std::cout << "HUman position: " << p_h << std::endl;
        std::cout << "HUman velocity: " << pd_h << std::endl;
        std::cout << "HUman acceleration: " << pdd_h << std::endl;

                   }
    /* 
            
            // =================================================================
            // LOOP w4: PFL Velocities (outer loop)
            // =================================================================
            for (int w4 = 0; w4 < numberPFL; ++w4) {
                double velocity_PFL = params.vel_PFL[w4];
                
                // =============================================================
                // LOOP w3: Qv Values (inner loop)
                // =============================================================
                for (int w3 = 0; w3 < number_QpQv; ++w3) {
                    double Qv = params.Qvs[w3];
                    
                    // Initialize simulation state
                    Vector6 qddot_real = Vector6::Zero();
                    Vector6 qdot_real = traj.qd.col(0);
                    Vector6 q_real = traj.q.col(0);
                    Vector3 p_real = p_r[0];
                    Vector3 v_real = pd_r[0];
                    
                    std::vector<double> v_real_module;
                    std::vector<Vector6> qddot_real_history;
                    
                    bool collision = false;
                    int collision_counter = 0;
                    int arrival_step = number_time_points_complete;
                    int reference_time = 0;
                    int failure_flag = 0;
                    
                    // Main control loop
                    for (int i = 0; i < number_time_points_complete - 1; ++i) {
                        if (!collision) {
                            // Compute safety distance delta
                            double delta_safety = 0.1 + pd_h.norm() * params.stopping_time;
                            
                            // PFL velocity scaling term (from MATLAB code)
                            double velocity_term = -( -(delta_safety / params.stopping_time) + velocity_PFL ) * params.stopping_time;
                            
                            // Call SSMPFL function (Appendix D)
                            SSMPFLResult res = SSMPFL(
                                robot,
                                params.joint_limits,
                                dt,
                                params.stopping_time,
                                q_real,
                                qdot_real,
                                p_r[reference_time + 1],
                                pd_r[reference_time + 1],
                                qddot_real,
                                p_h,
                                pd_h,
                                delta_safety,
                                traj.q.col(reference_time + 1),
                                traj.qd.col(reference_time + 1),
                                Qv,
                                velocity_term
                            );
                            
                            // Update state
                            qddot_real = res.qddot;
                            qdot_real = res.qdot_next;
                            q_real = res.q_next;
                            p_real = res.p_next;
                            v_real = res.v_next;
                            
                            v_real_module.push_back(v_real.norm());
                            qddot_real_history.push_back(qddot_real);
                            
                            // Increment reference trajectory time
                            reference_time++;
                            
                            // Check if optimization succeeded
                            if (res.flag == 1) {
                                // Check collision with human (distance <= 0.1m)
                                if ((p_real - p_h).norm() <= 0.1) {
                                    R_STOP[w1][w2][w3][w4] += 1.0;
                                    collision = true;
                                    collision_counter = 0;
                                    std::cout << "    Qv=" << Qv << ", PFL=" << velocity_PFL << " -> Collision at step " << i << std::endl;
                                }
                                
                                // Check if robot reached goal position
                                if ((p_real - p_rf).norm() <= 0.005) {
                                    arrival_step = i;
                                    std::cout << "    Qv=" << Qv << ", PFL=" << velocity_PFL << " -> Arrived at t=" << (i * dt) << "s" << std::endl;
                                    break;
                                }
                            } else {
                                // Optimization failed
                                failure_flag = 1;
                                std::cout << "    Qv=" << Qv << ", PFL=" << velocity_PFL << " -> Optimization failed" << std::endl;
                                break;
                            }
                        } else {
                            // Collision recovery phase
                            collision_counter++;
                            qddot_real.setZero();
                            qdot_real.setZero();
                            q_real.setZero();
                            p_real.setZero();
                            v_real.setZero();
                            v_real_module.push_back(0.0);
                            
                            if (collision_counter > params.pause_after_collision * params.computational_frequency) {
                                collision = false;
                                reference_time = 0;
                            }
                        }
                    }
                    
                    // =============================================================
                    // Store Results
                    // =============================================================
                    if (failure_flag == 1) {
                        T_TIME[w1][w2][w3][w4] = std::nan("");
                        v_real_rms[w1][w2][w3][w4] = std::nan("");
                        R_IDLE[w1][w2][w3][w4] = std::nan("");
                    } else {
                        if (arrival_step < number_time_points_complete) {
                            T_TIME[w1][w2][w3][w4] = arrival_step * dt;
                            
                            if (!v_real_module.empty()) {
                                v_real_rms[w1][w2][w3][w4] = math::rms(v_real_module);
                            }
                            
                            if (T_TIME[w1][w2][w3][w4] > 1e-6) {
                                R_IDLE[w1][w2][w3][w4] = R_STOP[w1][w2][w3][w4] * params.pause_after_collision / T_TIME[w1][w2][w3][w4];
                            }
                        } else {
                            T_TIME[w1][w2][w3][w4] = std::nan("");
                            v_real_rms[w1][w2][w3][w4] = std::nan("");
                            R_IDLE[w1][w2][w3][w4] = std::nan("");
                        }
                    }
                    
                    // Store original time (same for all Qv, PFL combinations)
                    T_TIME_ORIGINAL[w1][w2][w3][w4] = T_TIME_ORIGINAL_val;
                }
            }
        }
    }
    
    // =========================================================================
    // PLOT RESULTS (MATLAB: "Plot of the results")
    // =========================================================================
    std::cout << "\n=== Experiment Complete ===" << std::endl;
    std::cout << "Results stored in T_TIME, v_real_rms, R_STOP, R_IDLE matrices." << std::endl;
    
    // Example: Compute mean ratios for surface plots
    std::vector<std::vector<double>> mean_v_rms_ratio(number_QpQv, std::vector<double>(numberPFL, 0.0));
    std::vector<std::vector<double>> mean_T_TIME_ratio(number_QpQv, std::vector<double>(numberPFL, 0.0));
    std::vector<std::vector<double>> mean_R_STOP(number_QpQv, std::vector<double>(numberPFL, 0.0));
    std::vector<std::vector<double>> mean_R_IDLE(number_QpQv, std::vector<double>(numberPFL, 0.0));
    
    int valid_count = 0;
    for (int w3 = 0; w3 < number_QpQv; ++w3) {
        for (int w4 = 0; w4 < numberPFL; ++w4) {
            double sum_v = 0.0, sum_T = 0.0, sum_R_STOP = 0.0, sum_R_IDLE = 0.0;
            int count = 0;
            
            for (int w1 = 0; w1 < params.number_trajectories; ++w1) {
                for (int w2 = 0; w2 < params.number_intrusions_per_trajectory; ++w2) {
                    if (!std::isnan(v_real_rms[w1][w2][w3][w4])) {
                        double v_nominal = 1.0; // Use actual v_rms_nominal[w1][w2] in production
                        sum_v += v_nominal / v_real_rms[w1][w2][w3][w4];
                        
                        if (T_TIME_ORIGINAL[w1][w2][w3][w4] > 1e-6) {
                            sum_T += T_TIME[w1][w2][w3][w4] / T_TIME_ORIGINAL[w1][w2][w3][w4];
                        }
                        
                        sum_R_STOP += R_STOP[w1][w2][w3][w4];
                        sum_R_IDLE += R_IDLE[w1][w2][w3][w4];
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                mean_v_rms_ratio[w3][w4] = sum_v / count;
                mean_T_TIME_ratio[w3][w4] = sum_T / count;
                mean_R_STOP[w3][w4] = sum_R_STOP / count;
                mean_R_IDLE[w3][w4] = sum_R_IDLE / count;
            }
            
            std::cout << "Qv=" << params.Qvs[w3] << ", PFL=" << params.vel_PFL[w4] 
                      << " -> V-SMOOTHNESS=" << mean_v_rms_ratio[w3][w4]
                      << ", E-TIME=" << mean_T_TIME_ratio[w3][w4]
                      << ", R-STOP=" << mean_R_STOP[w3][w4] << std::endl;
        }
    }*/
    
    return 0;
}



















// ─────────────────────────────────────────────────────────────────────────────
// Main loop implementing chosen strategy
// ─────────────────────────────────────────────────────────────────────────────
int task_engine(
        std::vector<std::unique_ptr<DataTransmitter>>& transmitters, 
        RobotModel robot,
        int elapsed_ms, 
        const Eigen::VectorXd q_r, 
        const Eigen::VectorXd qd_r, 
        const Eigen::VectorXd qdd_r,
        Eigen::VectorXd& p_h, 
        Eigen::VectorXd& pd_h, 
        Eigen::VectorXd& pdd_h) {
    std::vector<Eigen::Vector3d> skeleton = json_to_keypoints(transmitters[0]->receive_data()[0]);
    std::optional<DistanceResult> dist = human_to_robot_distance(skeleton, robot, q_r);

    if (!dist) return 1;
    else std::cout << "Minimum distance between robot and skeleton: " << dist->length << std::endl;

    std::vector<nlohmann::json> payload;
    payload.push_back(std::vector<std::array<double, 3>>{{0, 0, 0}});
    payload.push_back(q_r);
    payload.push_back(std::vector<int>{});
    transmitters[1]->send_data(payload);

    payload.clear();
    payload.push_back(dist->c_h);
    payload.push_back(dist->c_r);
    transmitters[2]->send_data(payload);

    double loop_duration = 0.001 * static_cast<double>(elapsed_ms);
    Eigen::VectorXd p_h_prev = p_h;
    Eigen::VectorXd pd_h_prev = pd_h;
    p_h = dist->c_h;
    pd_h = (p_h - p_h_prev)/loop_duration;
    pdd_h = (pd_h - pd_h_prev)/loop_duration;

    SSM_PFL_escape(robot, q_r, qd_r, qdd_r, p_h, pd_h, pdd_h);
    
    return 0;
};


// ─────────────────────────────────────────────────────────────────────────────
// Load trajectory
// ─────────────────────────────────────────────────────────────────────────────
std::optional<Trajectory> load_trajectory(int n_traj, std::string c_dir) {
    std::string trajectory_path = c_dir + "src/trajectories/test" + std::to_string(n_traj) + "/";
    std::ifstream f(trajectory_path);
    try {
        if (!f) throw 1;
    }
    catch (int err) {
        std::cerr << "Error: cannot open '" + trajectory_path + "'" << std::endl;
        return std::nullopt;
    }

    std::vector<std::array<double, 7>> traj_low = load_trajectory_CSV(trajectory_path + "q_ref.csv");
    std::vector<double> t_low = load_timestamps_CSV(trajectory_path + "t_ref.csv");
    Trajectory traj_high = interpolate_to_1kHz_full(traj_low, t_low);
    save_trajectory_CSV(trajectory_path + "q.csv",  traj_high.q);
    save_trajectory_CSV(trajectory_path + "qd.csv",  traj_high.qd);
    save_trajectory_CSV(trajectory_path + "qdd.csv",  traj_high.qdd);

    return traj_high;
}


// ─────────────────────────────────────────────────────────────────────────────
// Execute task
// ─────────────────────────────────────────────────────────────────────────────
int execute_task (int n_traj, std::string c_dir="") {
    std::vector<std::unique_ptr<DataTransmitter>> transmitters;
    transmitters.reserve(3);
    transmitters.push_back(std::make_unique<DataTransmitter>(DataTransmitter::Mode::Receiver, 10, "MERGED"));
    transmitters.push_back(std::make_unique<DataTransmitter>(DataTransmitter::Mode::Sender, 12, "ROBOT"));
    transmitters.push_back(std::make_unique<DataTransmitter>(DataTransmitter::Mode::Sender, 13, "DISTANCE"));

    auto traj = load_trajectory(n_traj, c_dir);
    if (!traj) return 1;

    const std::string urdf_path = c_dir + "/src/urdf/panda.urdf";
    RobotModel robot(urdf_path);

    // ── Definition of human collision point ──────────────────────────────────────
    Eigen::VectorXd q(Eigen::Map<Eigen::VectorXd>((*traj).q[0].data(), (*traj).q[0].size()));
    std::vector<Eigen::Vector3d> skeleton = json_to_keypoints(transmitters[0]->receive_data()[0]);
    std::optional<DistanceResult> dist = human_to_robot_distance(skeleton, robot, q);
    Eigen::VectorXd p_h = dist->c_h;
    Eigen::VectorXd pd_h = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd pdd_h = Eigen::VectorXd::Zero(3);

    const double rate_hz = 16.0;
    const auto period = std::chrono::duration<double>(1.0 / rate_hz);
    auto next_time = std::chrono::steady_clock::now();
    auto loop_start = std::chrono::steady_clock::now();
    
    // ── Task engine ──────────────────────────────────────────────────────────────
    while (running) {
        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        int elapsed_ms = static_cast<int>(std::round(std::chrono::duration<double>(elapsed).count() * 1000));
        if (elapsed_ms < traj->q.size()) {
            Eigen::VectorXd q_r(Eigen::Map<Eigen::VectorXd>((*traj).q[elapsed_ms].data(), (*traj).q[elapsed_ms].size()));
            Eigen::VectorXd qd_r(Eigen::Map<Eigen::VectorXd>((*traj).qd[elapsed_ms].data(), (*traj).qd[elapsed_ms].size()));
            Eigen::VectorXd qdd_r(Eigen::Map<Eigen::VectorXd>((*traj).qdd[elapsed_ms].data(), (*traj).qdd[elapsed_ms].size()));
            
            task_engine(transmitters, robot, elapsed_ms, q_r, qd_r, qdd_r, p_h, pd_h, pdd_h);
        }
        else {
            loop_start = std::chrono::steady_clock::now();
        }
        
        next_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_time);
    }

    for (auto &transmitter : transmitters) {
        transmitter->shutdown();
    }

    return 0;
}


// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    int n_traj = 0;
    std::string path;
    if (argc > 1) {
        try {
            std::string arg(argv[1]);
            const char* begin = arg.data();
            const char* end   = arg.data() + arg.size();
            auto [ptr, ec] = std::from_chars(begin, end, n_traj);
            if (ec != std::errc{} || ptr != end)
                throw 1;
        }
        catch (int err) {
            std::cerr << "Error: argument must be a valid integer." << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << "Error: argument required." << std::endl;
        return 1;
    }
    if (argc > 2) {
        std::string c_dir(argv[2]);
        path = c_dir + "/";
    } 
    else {
        std::string c_dir(get_current_dir_name());
        path = c_dir + "/../";
    }

    std::signal(SIGINT, signal_handler);

    if (execute_task(n_traj, path)) return 1;
    else printf("Exiting cleanly...\n");
    
    return 0;
}
