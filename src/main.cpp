#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>
#include <iostream>
#include <string>
#include <vector>
#include <charconv>

#include "trajectory_utils.hpp"
#include "data_transmitter.hpp"
#include "utils.hpp"
#include "min_distance_calculation.hpp"
#include "kinematics.hpp"


const Eigen::Matrix<double, 7, 4> DH_default = []() {
    Eigen::Matrix<double, 7, 4> DH;
    const double pi = M_PI;
    DH << 0.0,    -pi/2.0,  0.0,     0.333,
          0.0,    -pi/2.0,  pi,      0.0,
          0.088,   pi/2.0,  pi,      0.316,
          0.088,   pi/2.0,  pi,      0.0,
          0.0,     pi/2.0,  pi,      0.384,
          0.088,   pi/2.0,  0.0,     0.0,
          0.0,     0.0,     0.0,     0.2;
    return DH;
}();


Eigen::Matrix<double, 6, Eigen::Dynamic> Jacobian(
    const Eigen::VectorXd& q,
    const Eigen::MatrixXd& DH,
    const Eigen::Vector3d& A_k)
{
    int n = static_cast<int>(q.size());
    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, n);
    J.setZero();

    Eigen::Vector3d z0(0.0, 0.0, 1.0);
    Eigen::Vector3d p0(0.0, 0.0, 0.0);

    Eigen::Matrix3d R_k = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d T_k = Eigen::Matrix4d::Identity();

    const double pi = M_PI;

    // First joint (index 0)
    {
        Eigen::Vector3d zj = z0;
        Eigen::Vector3d pj = p0;
        Eigen::Vector3d p = A_k;

        Eigen::Vector3d Jk = (zj.cross(p - pj));
        J.col(0).head(3) = Jk;
        J.col(0).tail(3) = zj;
    }

    for (int i = 0; i < n - 1; ++i) {
        double a     = DH(i, 0);
        double alpha = DH(i, 1);
        double d     = DH(i, 2);
        double theta_offset = DH(i, 3);

        double theta = q(i) + theta_offset;

        double ct = std::cos(theta);
        double st = std::sin(theta);
        double ca = std::cos(alpha);
        double sa = std::sin(alpha);

        Eigen::Matrix3d R;
        R << ct, -st * ca,  st * sa,
             st,  ct * ca, -ct * sa,
             0,   sa,       ca;

        Eigen::Matrix4d T;
        T << ct, -st * ca,  st * sa,  a * ct,
             st,  ct * ca, -ct * sa,  a * st,
             0,   sa,       ca,       d,
             0,   0,        0,        1;

        R_k = R_k * R;
        T_k = T_k * T;

        Eigen::Vector3d zj = R_k * z0;
        Eigen::Vector3d pj = T_k.block<3, 1>(0, 3);
        Eigen::Vector3d p = A_k;

        Eigen::Vector3d Jk = zj.cross(p - pj);
        J.col(i + 1).head(3) = Jk;
        J.col(i + 1).tail(3) = zj;
    }

    return J;
}

Eigen::Vector3d compute_capsule_end_point(
    const Eigen::Matrix<double, 7, 4>& DH,
    const Eigen::Vector<double, 7>& q,
    int k)
{
    Eigen::Matrix4d T_k = Eigen::Matrix4d::Identity();
    const double pi = M_PI;

    for (int j = 0; j < k; ++j) {
        double a     = DH(j, 0);
        double alpha = DH(j, 1);
        double d     = DH(j, 2);
        double theta_offset = DH(j, 3);

        double theta = q(j) + theta_offset;

        double ct = std::cos(theta);
        double st = std::sin(theta);
        double ca = std::cos(alpha);
        double sa = std::sin(alpha);

        Eigen::Matrix4d T;
        T << ct, -st * ca,  st * sa,  a * ct,
             st,  ct * ca, -ct * sa,  a * st,
             0,   sa,       ca,       d,
             0,   0,        0,        1;

        T_k = T_k * T;
    }

    return Eigen::Vector3d(T_k(0, 3), T_k(1, 3), T_k(2, 3));
}


// Global flag, set by the signal handler
std::atomic<bool> running{true};

void signal_handler(int signum) {
    (void)signum;
    running = false;
}


// ─────────────────────────────────────────────────────────────────────────────
// Main loop implementing SSM strategy
// ─────────────────────────────────────────────────────────────────────────────
int SSM_strategy(DataTransmitter& dtr, DataTransmitter& dts, std::vector<double> q) {
    std::vector<nlohmann::json> payload;
    payload.push_back(std::vector<std::array<double, 3>>{{0, 0, 0}});
    payload.push_back(q);
    payload.push_back(std::vector<int>{});
    dts.send_skeleton_data(payload);

    std::vector<Eigen::Vector3d> skeleton = json_to_keypoints(dtr.receive_skeleton_data()[0]);

    std::cout << q[0] << q[1] << q[2] << q[3] << q[4] << q[5] << q[6] << std::endl;

    // Eigen::Vector<double, 7> q_eigen = Eigen::Map<Eigen::Vector<double, 7>>(q.data());
    // Eigen::Vector3d A_3 = compute_capsule_end_point(DH, q_eigen, 3);
// 
    // Eigen::Matrix<double, 6, 1> J5 = Jacobian(q_eigen.head(3), DH.topRows(3), A_3);
// 
    // std::cout << "Jacobian: " << J5[0] << J5[1] << J5[2] << J5[3] << J5[4] << J5[5] << std::endl;


    return 0;
};


// ─────────────────────────────────────────────────────────────────────────────
// Execute trajectory
// ─────────────────────────────────────────────────────────────────────────────
int load_trajectory (int n_traj, std::string c_dir="") {
    DataTransmitter dtr = DataTransmitter(DataTransmitter::Mode::Receiver, 10, "MERGED");
    DataTransmitter dts = DataTransmitter(DataTransmitter::Mode::Sender, 12, "ROBOT");

    std::signal(SIGINT, signal_handler);

    std::string trajectory_path = c_dir + "src/trajectories/test" + std::to_string(n_traj) + "/";
    printf("Loading trajectory %d from path: %s\n", n_traj, trajectory_path.c_str());
    std::ifstream f(trajectory_path);
    try {
        if (!f) throw 1;
    }
    catch (int err) {
        std::cerr << "Error: cannot open '" + trajectory_path + "'" << std::endl;
        return 1;
    }

    std::vector<std::array<double, 7>> traj_low = load_trajectory_CSV(trajectory_path + "q.csv");
    std::vector<double> t_low = load_timestamps_CSV(trajectory_path + "t.csv");
    auto traj_high = interpolate_to_1kHz(traj_low, t_low);
    save_trajectory_CSV(trajectory_path + "q_1kHz.csv",  traj_high);

    std::cout << std::endl << "── Starting trajectory ──────────────────────────────────────────────────────" << std::endl << std::endl;

    const double rate_hz = 5.0;
    const auto period = std::chrono::duration<double>(1.0 / rate_hz);
    auto next_time = std::chrono::steady_clock::now();
    auto loop_start = std::chrono::steady_clock::now();
    while (running) {
        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        int elapsed_ms = static_cast<int>(std::round(std::chrono::duration<double>(elapsed).count() * 1000));
        if (elapsed_ms < traj_high.size()) {
            std::vector<double> q(traj_high[elapsed_ms].begin(), traj_high[elapsed_ms].end());


            // ── SSM strategy ─────────────────────────────────────────────────────────────
            SSM_strategy(dtr, dts, q);
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

    if (load_trajectory(n_traj, path)) {
        return 1;
    }
    else {
        printf("Exiting cleanly...\n");
    }
    
    return 0;
}
