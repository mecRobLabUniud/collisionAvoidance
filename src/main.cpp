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
#include "panda_jacobian.hpp"

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







    const std::string urdf_path = c_dir + "/src/urdf/panda.urdf";

    






    const double rate_hz = 5.0;
    const auto period = std::chrono::duration<double>(1.0 / rate_hz);
    auto next_time = std::chrono::steady_clock::now();
    auto loop_start = std::chrono::steady_clock::now();
    while (running) {
        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        int elapsed_ms = static_cast<int>(std::round(std::chrono::duration<double>(elapsed).count() * 1000));
        if (elapsed_ms < traj_high.size()) {
            std::vector<double> q(traj_high[elapsed_ms].begin(), traj_high[elapsed_ms].end());






            PandaJacobian panda_jacobian(urdf_path, "panda_link8");

            Eigen::VectorXd q_vec(Eigen::Map<Eigen::VectorXd>(q.data(), q.size()));

            Eigen::MatrixXd J = panda_jacobian.ComputeJacobian(q_vec);
            std::cout << "Jacobian (6x7):\n" << J << std::endl;





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
