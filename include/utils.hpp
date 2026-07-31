#pragma once
#include <Eigen/Core>
#include <array>
#include <vector>
#include <nlohmann/json.hpp>

double json_to_double(const nlohmann::json& v) {
    if (v.is_null()) return std::numeric_limits<double>::quiet_NaN();
    return v.get<double>();
}


std::vector<Eigen::Vector3d> json_to_keypoints(const nlohmann::json& arr) {
    std::vector<Eigen::Vector3d> out;
    out.reserve(arr.size());
    for (const auto& p : arr) {
        out.push_back({ json_to_double(p[0]), json_to_double(p[1]), json_to_double(p[2]) });
    }
    return out;
}