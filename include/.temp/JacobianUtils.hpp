#pragma once
#include "RobotModel.hpp"

// -----------------------------------------------------------------------
// Appendix J: "derivative_Jacobian.m"
//
// function J_dot = derivative_Jacobian(robot, configuration, endeffectorname)
//     J  = geometricJacobian(robot, configuration, endeffectorname);
//     configuration_p = configuration + 0.0000001;
//     Jp = geometricJacobian(robot, configuration_p, endeffectorname);
//     J_dot = (Jp - J) / 0.0000001;
// end
//
// NOTE (preserved as-is from the source): this perturbs *every* joint
// simultaneously by the same epsilon rather than differentiating w.r.t.
// each joint separately and contracting with qdot. It is a crude forward-
// difference approximation of the Jacobian's time derivative along the
// direction "all joints moving together", not a true dJ/dq. Kept exactly
// as in the thesis code for fidelity; consider replacing with an
// analytical Jacobian derivative or a proper per-joint finite-difference
// (sum_i dJ/dq_i * qdot_i) if you need accuracy.
// -----------------------------------------------------------------------
inline Eigen::MatrixXd derivativeJacobian(const RobotModel& robot,
                                           const Eigen::VectorXd& configuration,
                                           const std::string& endEffectorName) {
    constexpr double eps = 0.0000001;

    Eigen::MatrixXd J = robot.geometricJacobian(configuration, endEffectorName);

    Eigen::VectorXd configuration_p = configuration.array() + eps;
    Eigen::MatrixXd Jp = robot.geometricJacobian(configuration_p, endEffectorName);

    return (Jp - J) / eps;
}
