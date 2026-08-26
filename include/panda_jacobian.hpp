// panda_jacobian.hpp
#pragma once

#include <string>

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

class PandaJacobian {
 public:
  explicit PandaJacobian(const std::string& urdf_path,
                          const std::string& ee_frame_name = "panda_link8");

  // Computes the 6xN geometric Jacobian (linear + angular) at the given
  // joint configuration, expressed in the LOCAL_WORLD_ALIGNED frame.
  Eigen::MatrixXd ComputeJacobian(const Eigen::VectorXd& q);

 private:
  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex ee_frame_id_;
};