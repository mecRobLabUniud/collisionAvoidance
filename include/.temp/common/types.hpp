#pragma once
#include <Eigen/Core>
#include <vector>

namespace appendices {

using Vector3  = Eigen::Vector3d;
using Vector6  = Eigen::Vector6d;
using Matrix3  = Eigen::Matrix3d;
using Matrix6  = Eigen::Matrix6d;
using Matrix36 = Eigen::Matrix<double, 3, 6>;
using Matrix66 = Eigen::Matrix6d;
using Transform4 = Eigen::Matrix4d;

struct JointLimits {
    Vector6 q_min, q_max;
    Vector6 qd_min, qd_max;
    Vector6 qdd_min, qdd_max;
};

} // namespace appendices