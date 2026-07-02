/*
░█▀▄░█░█░█░░░█▀█░░░█▀▀░█░█░█▀█░█░░░█░█░█▀█░▀█▀░▀█▀░█▀█░█▀█
░█▀▄░█░█░█░░░█▀█░░░█▀▀░▀▄▀░█▀█░█░░░█░█░█▀█░░█░░░█░░█░█░█░█
░▀░▀░▀▀▀░▀▀▀░▀░▀░░░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀
*/

#include <string>
#include <iostream>
#include <cmath>
#include <array>
#include <optional>
#include <algorithm>
#include <vector>

// #include "skeleton_receiver.h"
#include "rula_score_computation.h"
#include "data_transmitter.hpp"


// ─────────────────────────────────────────────────────────────────────────────
// Skeleton parser from string
// ─────────────────────────────────────────────────────────────────────────────
std::array<double, 3> parseTriplet(const std::string& s, size_t& pos) {
    // Find opening '['
    pos = s.find('[', pos);
    if (pos == std::string::npos) throw std::runtime_error("Expected '['");
    pos++; // skip '['

    std::array<double, 3> triplet;
    for (int i = 0; i < 3; i++) {
        // Skip whitespace and commas
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',')) pos++;

        // Extract token until ',' or ']'
        size_t end = pos;
        while (end < s.size() && s[end] != ',' && s[end] != ']') end++;

        std::string token = s.substr(pos, end - pos);
        // Trim whitespace
        token.erase(0, token.find_first_not_of(' '));
        token.erase(token.find_last_not_of(' ') + 1);

        if (token == "NaN" || token == "nan") {
            triplet[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            triplet[i] = std::stod(token);
        }
        pos = end;
    }

    // Find closing ']'
    pos = s.find(']', pos);
    if (pos == std::string::npos) throw std::runtime_error("Expected ']'");
    pos++; // skip ']'

    return triplet;
}

std::vector<std::array<double, 3>> parseKeypointList(const std::string& s, size_t& pos) {
    // Find opening '[' of the outer list
    pos = s.find('[', pos);
    if (pos == std::string::npos) throw std::runtime_error("Expected outer '['");
    pos++; // skip outer '['

    std::vector<std::array<double, 3>> keypoints;

    while (pos < s.size()) {
        // Skip whitespace and commas
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',' || s[pos] == '\n')) pos++;

        if (s[pos] == ']') { pos++; break; } // end of outer list
        if (s[pos] == '[') {
            keypoints.push_back(parseTriplet(s, pos));
        } else {
            pos++;
        }
    }

    return keypoints;
}

std::vector<std::array<double, 3>> parseMergedString(const std::string& input, std::string& label) {
    size_t pos = 0;

    // Extract label (everything before the first ';')
    size_t semicolon = input.find(';');
    if (semicolon == std::string::npos) throw std::runtime_error("Expected ';' separator");
    label = input.substr(0, semicolon);
    label.erase(label.find_last_not_of(' ') + 1); // trim trailing space

    pos = semicolon + 1;

    return parseKeypointList(input, pos);
}


// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    AdjustmentFlags flags;
    flags.isRepeated    = false;
    flags.forceScoreA   = 0;
    flags.forceScoreB   = 0;
    
    auto t0 = std::chrono::steady_clock::now();

    DataTransmitter dtr = DataTransmitter(DataTransmitter::Mode::Receiver, 10, "MERGED", 7000);
    DataTransmitter dts = DataTransmitter(DataTransmitter::Mode::Sender, 11, "RULA", 7000);
    // DataTransmitter dtr_rula = DataTransmitter(DataTransmitter::Mode::Receiver, 11, "RULA", 7000);
    while (true) {
        if (true) {
            auto start = std::chrono::steady_clock::now();

            auto skeleton = dtr.receive_skeleton_data().first;
            RULAResult result_R = computeRULA(skeleton, flags, 'R', false);
            RULAResult result_L = computeRULA(skeleton, flags, 'L', false);

            // result_R.print();

            auto end = std::chrono::steady_clock::now();
            // Cast to whatever unit you need
            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            // std::cout << "Score R: " << result_R.grandScore << "  -  ";
            // std::cout << "Score L: " << result_L.grandScore << "  -  ";
            // std::cout << "Elapsed time: " << elapsed_ms << "ms" << "\r";

            std::array<int, 2> rula_score = {result_R.grandScore, result_L.grandScore};
            dts.send_rula_score(rula_score);
            // auto score = dtr_rula.receive_rula_score();

            // std::cout << "SCORE: " << score[0] << ", " << score[1] << "\r";
        }

        auto t1 = std::chrono::steady_clock::now();
        double elapsed_tot = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return 0;
}



// int main()
// {
//     Skeleton kp = {
//         { 0.32561093,  0.58366576,  0.88077476},
//         { 0.25705159,  0.75563931,  0.69358229},
//         { 0.33072013,  0.43864895,  0.65664771},
//         { 0.21092594,  1.059997  ,  0.64530905},
//         { 0.33662338,  0.44507781,  0.38093787},
//         { 0.22832004,  1.31962699,  0.68518367},
//         { 0.36859576,  0.38922919,  0.10281177},
//         { 0.29388586,  0.59714413,  0.675115  },
//         { 0.3453726 ,  0.6424219 ,  0.14056365},
//         { 0.31562263,  0.73525941,  0.15264951},
//         { 0.37512257,  0.54958439,  0.1284778 },
//         { 0.27161685,  0.73644812, -0.27857493},
//         { 0.2757422 ,  0.50638936, -0.36591824},
//         { 0.27161685,  0.73644812, -0.6857493},
//         { 0.2757422 ,  0.50638936, -0.6591824},
//     };
// 
// 
//     AdjustmentFlags flags;
//     flags.isRepeated    = false;   // task is repetitive
//     flags.forceScoreA   = 0;      // 2-10 kg load, intermittent
//     flags.forceScoreB   = 0;
// 
//     RULAResult resultR = computeRULA(kp, flags, 'R', false);
//     resultR.print();
// 
//     RULAResult resultL = computeRULA(kp, flags, 'L', false);
//     resultL.print();
// 
//     return 0;
// }