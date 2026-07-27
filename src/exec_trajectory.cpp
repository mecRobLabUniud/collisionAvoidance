#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>
#include <iostream>
#include <string>
#include <vector>

#include "load_trajectory.hpp"
#include "data_transmitter.hpp"

// Global flag, set by the signal handler
std::atomic<bool> running{true};

void signalHandler(int signum) {
    (void)signum;
    running = false;
}

int main() {
    std::signal(SIGINT, signalHandler);  // Ctrl+C

    DataTransmitter dts = DataTransmitter(DataTransmitter::Mode::Sender, 12, "ROBOT", 7000);
    DataTransmitter dtr = DataTransmitter(DataTransmitter::Mode::Receiver, 12, "ROBOT", 7000);

    /* if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.bin> <output.bin> <source_rate_hz>\n";
        return 1;
    }

    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    double      source_rate = std::stod(argv[3]);

    auto traj_low  = loadBin(input_path);
    std::cout << "Loaded " << traj_low.size()
              << " waypoints @ " << source_rate << " Hz\n"; */

    std::string trajectory_path = "../src/trajectories/test/";
    int columns = 7;

    std::vector<std::array<double, 7>> traj_low = loadTrajectoryCSV(trajectory_path + "q.csv");
    std::vector<double> t_low = loadTimestampsCSV(trajectory_path + "t.csv");
    std::cout << "Loaded " << traj_low.size() << "\n";

    auto traj_high = interpolateTo1kHz(traj_low, t_low);
    std::cout << "Interpolated to " << traj_high.size()
              << " waypoints @ " << 1000.0 << " Hz\n";

    saveTrajectoryCSV(trajectory_path + "q_1kHz.csv",  traj_high);

    // validateTrajectory(traj_high);
    // saveBin(output_path, traj_high);
    // std::cout << "Saved " << output_path << "\n";



    const double rate_hz = 30.0; // <-- control your loop rate here
    const auto period = std::chrono::duration<double>(1.0 / rate_hz);

    auto next_time = std::chrono::steady_clock::now();
    auto loop_start = std::chrono::steady_clock::now();

    while (running)
    {
        

        auto elapsed = std::chrono::steady_clock::now() - loop_start;
        int elapsed_ms = static_cast<int>(std::round(std::chrono::duration<double>(elapsed).count() * 1000));

        const std::vector<std::array<double, 3>> p = {{0, 0, 0}};
        const std::vector<int> _ = {};

        if (elapsed_ms <= traj_high.size()) {
            printf("t: %i ms - q: %.6f %.6f %.6f %.6f %.6f %.6f %.6f \n", elapsed_ms, 
                traj_high[elapsed_ms][0], traj_high[elapsed_ms][1], traj_high[elapsed_ms][2], traj_high[elapsed_ms][3], 
                traj_high[elapsed_ms][4], traj_high[elapsed_ms][5], traj_high[elapsed_ms][6]);

            std::vector<double> q(traj_high[elapsed_ms].begin(), traj_high[elapsed_ms].end());

            std::vector<nlohmann::json> payload;
            payload.push_back(p);
            payload.push_back(q);
            payload.push_back(_);

            dts.send_skeleton_data(payload);
        }
        

        next_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_time);
    }

    printf("Exiting cleanly...\n");
    return 0;
}
