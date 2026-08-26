// panda_jacobian.cpp
#include "panda_jacobian.hpp"

#include <stdexcept>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

PandaJacobian::PandaJacobian(const std::string& urdf_path,
                              const std::string& ee_frame_name) {
  pinocchio::urdf::buildModel(urdf_path, model_);
  data_ = pinocchio::Data(model_);

  if (!model_.existFrame(ee_frame_name)) {
    throw std::runtime_error("End-effector frame not found in URDF: " +
                              ee_frame_name);
  }
  ee_frame_id_ = model_.getFrameId(ee_frame_name);
}

Eigen::MatrixXd PandaJacobian::ComputeJacobian(const Eigen::VectorXd& q) {
  if (q.size() != model_.nq) {
    throw std::runtime_error("Joint vector size does not match model DOF");
  }

  // Forward kinematics must be computed before the Jacobian.
  pinocchio::forwardKinematics(model_, data_, q);
  pinocchio::updateFramePlacements(model_, data_);

  Eigen::MatrixXd J(6, model_.nv);
  J.setZero();

  pinocchio::computeFrameJacobian(
      model_, data_, q, ee_frame_id_,
      pinocchio::LOCAL_WORLD_ALIGNED, J);

  return J;
}