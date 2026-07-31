#pragma once
#include <Eigen/Dense>
#include <string>

// -----------------------------------------------------------------------
// Abstract interface standing in for MATLAB's rigidBodyTree object.
// The original MATLAB code calls getTransform(robot, q, linkName) and
// geometricJacobian(robot, q, linkName) for a handful of named links:
//
//   "base_link", "shoulder_link", "forearm_link",
//   "wrist_1_link", "wrist_2_link", "tool0"
//
// (These are the UR5e link names used in the thesis; swap in your own
// robot's link names if you plug in a different manipulator/model.)
//
// Implement this interface with your own forward kinematics (DH, URDF via
// KDL/Pinocchio, etc.). Nothing else in this translation depends on a
// specific kinematic representation.
// -----------------------------------------------------------------------
class RobotModel {
public:
    virtual ~RobotModel() = default;

    // 4x4 homogeneous transform of `linkName` w.r.t. the base frame, at
    // joint configuration q. Equivalent to MATLAB's getTransform(robot,q,linkName).
    virtual Eigen::Matrix4d getTransform(const Eigen::VectorXd& q,
                                          const std::string& linkName) const = 0;

    // 6 x n geometric Jacobian of `linkName`, MATLAB Robotics System Toolbox
    // convention: rows 0-2 = angular velocity part, rows 3-5 = linear
    // velocity part. Equivalent to geometricJacobian(robot,q,linkName).
    virtual Eigen::MatrixXd geometricJacobian(const Eigen::VectorXd& q,
                                               const std::string& linkName) const = 0;

    virtual int numJoints() const = 0;
};

// Convenience: link names used throughout the SSM/PFL/escape functions.
namespace links {
    inline const std::string base     = "base_link";
    inline const std::string shoulder = "shoulder_link";
    inline const std::string forearm  = "forearm_link";
    inline const std::string wrist1   = "wrist_1_link";
    inline const std::string wrist2   = "wrist_2_link";
    inline const std::string tool0    = "tool0";
}

// Extracts the linear-velocity (3 x n) block from a 6 x n geometric
// Jacobian, i.e. MATLAB's J_general(4:6,:).
inline Eigen::MatrixXd linearBlock(const Eigen::MatrixXd& J6) {
    return J6.bottomRows(3);
}

// Extracts the translation (3x1) part of a 4x4 homogeneous transform,
// i.e. MATLAB's T(1:3,4).
inline Eigen::Vector3d translationOf(const Eigen::Matrix4d& T) {
    return T.block<3, 1>(0, 3);
}
