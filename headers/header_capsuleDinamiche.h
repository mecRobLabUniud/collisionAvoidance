#ifndef HEADER_CAPSULEDINAMICHE_H
#define HEADER_CAPSULEDINAMICHE_H

#include <Eigen/Dense>
#include <franka/model.h>
#include <franka/robot.h>
#include <array>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <casadi/casadi.hpp>
#include "examples_common.h"

// Pinocchio includes
#include "pinocchio/fwd.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/data.hpp"

// ============================================================================
// STRUTTURE DATI
// ============================================================================

struct CapsuleGeo {
    Eigen::Vector3d p_start;
    Eigen::Vector3d p_end;
    double radius;
};

struct DistanceResult {
    double distance;
    Eigen::Vector3d closest_point_1; // Robot
    Eigen::Vector3d closest_point_2; // Umano
};

struct DynamicSafetyParams {
    double T_reaction;              // Tempo reazione safety controller (es. 0.005s per il paper)
    double v_human_max;             // 1.6 m/s (ISO/TS 15066)
    double csi;                     // Errore sensori
    std::vector<double> rv;         // Raggi fisici robot [0.085, 0.085, 0.06, 0.065]
    
    Eigen::Matrix<double, 7, 1> tau_max; // Limiti coppia Franka
    Eigen::Matrix<double, 7, 1> tau_rate_max; // Limiti derivata coppia (Nm/s)
    
    // Pesi ottimizzazione CasADi
    double w0 = 1e6;    // Peso minimizzazione t_stop
    double w1 = 1e5;    // Peso continuità (penalizza discontinuità)
};

struct StopTimeResult {
    double t_stop;
    bool feasible;
    std::vector<Eigen::VectorXd> q_traj;
    std::vector<Eigen::VectorXd> qp_traj;
    std::vector<Eigen::VectorXd> qpp_traj;
};

// ============================================================================
// FUNZIONI PRINCIPALI
// ============================================================================

// Ottimizzazione tempo di stop con CasADi + Pinocchio (Dinamica)
StopTimeResult optimize_stop_time_casadi_hybrid(
    const Eigen::Matrix<double, 7, 1>& q_current,
    const Eigen::Matrix<double, 7, 1>& qp_current,
    const Eigen::Matrix<double, 7, 1>& qpp_current,
    double t_stop_prev,
    const DynamicSafetyParams& params,
    const franka::Model& model_franka,
    const franka::RobotState& state,
    const pinocchio::Model& pin_model
);

// Verifica vincoli coppia numericamente
bool check_torque_limits_numeric(
    const std::vector<Eigen::VectorXd>& q_traj,
    const std::vector<Eigen::VectorXd>& qp_traj,
    const std::vector<Eigen::VectorXd>& qpp_traj,
    const DynamicSafetyParams& params,
    const franka::Model& model,
    const franka::RobotState& state
);

// Calcola raggi capsule dinamiche robot (Aggiornato per supportare array di distanze)
std::array<double, 4> compute_dynamic_capsule_radii(
    const StopTimeResult& stop_result,
    const Eigen::Matrix<double, 7, 1>& q_current,
    const Eigen::Matrix<double, 7, 1>& qp_current,
    const std::array<DistanceResult, 4>& capsule_distances,
    const DynamicSafetyParams& params,
    const franka::Model& model,
    const franka::RobotState& state
);

// ============================================================================
// UTILITY
// ============================================================================

// Calcola coefficienti polinomio quintico
Eigen::MatrixXd compute_traj_poly5_coeffs(
    double qi, double qi_p, double qi_pp,
    double qf, double qf_p, double qf_pp,
    double duration
);

// Genera traiettoria quintica completa
void generate_quintic_trajectory(
    const Eigen::Matrix<double, 7, 1>& q_start,
    const Eigen::Matrix<double, 7, 1>& qp_start,
    const Eigen::Matrix<double, 7, 1>& qpp_start,
    const Eigen::Matrix<double, 7, 1>& q_end,
    const Eigen::Matrix<double, 7, 1>& qp_end,
    const Eigen::Matrix<double, 7, 1>& qpp_end,
    double duration,
    int n_samples,
    std::vector<Eigen::VectorXd>& q_out,
    std::vector<Eigen::VectorXd>& qp_out,
    std::vector<Eigen::VectorXd>& qpp_out
);

// Ottieni posizione frame
Eigen::Vector3d get_joint_position(
    const franka::Model& model,
    const franka::RobotState& state,
    franka::Frame frame
);

// Calcola Jacobiano geometrico
Eigen::MatrixXd compute_geometric_jacobian(
    const franka::Model& model,
    const franka::RobotState& state,
    int k
);

// Calcola velocità endpoint capsula
Eigen::Vector3d compute_capsule_endpoint_velocity(
    const Eigen::Vector3d& p_joint,
    const Eigen::Vector3d& omega_joint,
    const Eigen::Vector3d& p_endpoint,
    const Eigen::Vector3d& p_other_end,
    double rv
);

// Distanza tra segmenti
DistanceResult distance_to_segment(
    Eigen::Vector3d P1, Eigen::Vector3d Q1,
    Eigen::Vector3d P2, Eigen::Vector3d Q2
);

// Logging
void save_log_to_csv(
    const std::vector<franka::RobotState>& log_data,
    const std::string& filename
);

#endif // HEADER_CAPSULEDINAMICHE_H