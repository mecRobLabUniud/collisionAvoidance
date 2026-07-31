#pragma once
#include <Eigen/Core>
#include <array>
#include <vector>

struct DistanceResult {
    double length;
    Eigen::Vector3d C_h;
    Eigen::Vector3d C_r;
};

DistanceResult segm_to_segm_distance(
    const Eigen::Vector3d& P1,
    const Eigen::Vector3d& Q1,
    const Eigen::Vector3d& P2,
    const Eigen::Vector3d& Q2);


std::optional<DistanceResult> human_robot_distance(
    const std::array<Eigen::Vector3d, 15>& skeleton,
    const Eigen::Vector3d& P2,
    const Eigen::Vector3d& Q2);