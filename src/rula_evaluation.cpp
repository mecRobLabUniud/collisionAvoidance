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

            auto skeleton = dtr.receive_skeleton_data()[0].get<std::vector<std::array<double,3>>>();

            RULAResult result_R = computeRULA(skeleton, flags, 'R', false);
            RULAResult result_L = computeRULA(skeleton, flags, 'L', false);

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