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


// ─────────────────────────────────────────────────────────────────────────────
// Main loop implementing chosen strategy
// ─────────────────────────────────────────────────────────────────────────────
int task_engine(DataTransmitter& dtr, DataTransmitter& dts, RobotModel robot, std::vector<double> q_vec) {
    std::vector<Eigen::Vector3d> skeleton = json_to_keypoints(dtr.receive_skeleton_data()[0]);

    Eigen::VectorXd q(Eigen::Map<Eigen::VectorXd>(q_vec.data(), q_vec.size()));
    double HR_clearance = 0.1;
    bool flag = false;
    // for (const auto& point : skeleton) {
    //     if (point.hasNaN()) continue;
// 
    //     // SSMPFLResult res = SSMPFL(const RobotModel& robot,
    //     //                     const Eigen::MatrixXd& q_limits,    // n x 2 [min max]
    //     //                     const Eigen::MatrixXd& qdot_limits, // n x 2
    //     //                     const Eigen::MatrixXd& qddot_limits,// n x 2
    //     //                     double delta_t,
    //     //                     double stopping_time,
    //     //                     const Eigen::VectorXd& q_t,
    //     //                     const Eigen::VectorXd& qdot_t,
    //     //                     const Eigen::Vector3d& x_ref_tplusone,
    //     //                     const Eigen::Vector3d& xd_ref_tplusone,
    //     //                     const Eigen::VectorXd& /*qddot_suggestion*/, // unused in source too
    //     //                     Eigen::Vector3d ro,
    //     //                     const Eigen::Vector3d& vo,
    //     //                     double delta,
    //     //                     const Eigen::VectorXd& q_des,
    //     //                     const Eigen::VectorXd& /*qd_des*/, // unused in source too
    //     //                     double Qv)
    // }

    std::optional<DistanceResult> dist = human_to_robot_distance(skeleton, robot, q);

    if (dist) {
        std::cout << "Minimum distance between robot and skeleton: " << dist->length << std::endl;
    }

    std::vector<nlohmann::json> payload;
    payload.push_back(std::vector<std::array<double, 3>>{{0, 0, 0}});
    payload.push_back(q_vec);
    payload.push_back(std::vector<int>{});
    dts.send_skeleton_data(payload);
    
    return 0;
};


// ─────────────────────────────────────────────────────────────────────────────
// Load trajectory
// ─────────────────────────────────────────────────────────────────────────────
std::optional<std::vector<std::array<double, 7>>> load_trajectory(int n_traj, std::string c_dir) {
    std::string trajectory_path = c_dir + "src/trajectories/test" + std::to_string(n_traj) + "/";
    std::ifstream f(trajectory_path);
    try {
        if (!f) throw 1;
    }
    catch (int err) {
        std::cerr << "Error: cannot open '" + trajectory_path + "'" << std::endl;
        return std::nullopt;
    }

    std::vector<std::array<double, 7>> traj_low = load_trajectory_CSV(trajectory_path + "q.csv");
    std::vector<double> t_low = load_timestamps_CSV(trajectory_path + "t.csv");
    std::vector<std::array<double, 7>> traj_high = interpolate_to_1kHz(traj_low, t_low);
    save_trajectory_CSV(trajectory_path + "q_1kHz.csv",  traj_high);

    return traj_high;
}






/*
    const std::string urdf_path = c_dir + "/src/urdf/panda.urdf";

    RobotModel robot(urdf_path);

    Eigen::VectorXd q(7);
    q << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;  // Panda "ready" pose

    // Forward kinematics + pose lookup
    Eigen::Isometry3d ee_pose = robot.GetJointPose("panda_link0", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;
    ee_pose = robot.GetJointPose("panda_link1", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;
    ee_pose = robot.GetJointPose("panda_link2", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;
    ee_pose = robot.GetJointPose("panda_link3", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;      
    ee_pose = robot.GetJointPose("panda_link4", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;
    ee_pose = robot.GetJointPose("panda_link5", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;
    ee_pose = robot.GetJointPose("panda_link6", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;
    ee_pose = robot.GetJointPose("panda_link7", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl;     
    ee_pose = robot.GetJointPose("panda_link8", q);
    std::cout << "EE position: " << ee_pose.translation().transpose()
                << std::endl; 


    // Jacobian at same q
    Eigen::MatrixXd J = robot.ComputeJacobian("panda_link8", q);
    std::cout << "Jacobian (6x7):\n" << J << std::endl;

    // Inverse kinematics: try to reach a slightly perturbed target
    Eigen::Isometry3d target = ee_pose;
    target.translation().z() += 0.05;

    Eigen::VectorXd q_solution;
    bool ok = robot.ComputeIK("panda_link8", target, q, &q_solution);
    if (ok) {
        std::cout << "IK solution: " << q_solution.transpose() << std::endl;
    } else {
        std::cout << "IK did not converge" << std::endl;
    }
*/




// ─────────────────────────────────────────────────────────────────────────────
// Execute task
// ─────────────────────────────────────────────────────────────────────────────
int execute_task (int n_traj, std::string c_dir="") {
    DataTransmitter dtr = DataTransmitter(DataTransmitter::Mode::Receiver, 10, "MERGED");
    DataTransmitter dts = DataTransmitter(DataTransmitter::Mode::Sender, 12, "ROBOT");

    auto traj = load_trajectory(n_traj, c_dir);
    if (!traj) return 1;

    const double rate_hz = 5.0;
    const auto period = std::chrono::duration<double>(1.0 / rate_hz);
    auto next_time = std::chrono::steady_clock::now();
    auto loop_start = std::chrono::steady_clock::now();
    const std::string urdf_path = c_dir + "/src/urdf/panda.urdf";
    RobotModel robot(urdf_path);

    // ── Task engine ──────────────────────────────────────────────────────────────
    while (running) {
        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        int elapsed_ms = static_cast<int>(std::round(std::chrono::duration<double>(elapsed).count() * 1000));
        if (elapsed_ms < traj->size()) {
            std::vector<double> q((*traj)[elapsed_ms].begin(), (*traj)[elapsed_ms].end());
            task_engine(dtr, dts, robot, q);
        }
        else {
            loop_start = std::chrono::steady_clock::now();
        }
        
        next_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_time);
    }

    dtr.shutdown();
    dts.shutdown();

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
