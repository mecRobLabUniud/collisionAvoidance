#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>

#include <franka/robot.h>
#include <franka/model.h>
#include <franka/exception.h>
#include <Eigen/Dense>

#include "pinocchio/parsers/urdf.hpp"

#include "skeleton_zmq.h"
#include "examples_common.h"
#include "header_capsuleDinamiche.h"

// ============================================================================
// STRUTTURE PER MULTITHREADING
// ============================================================================
struct SharedRobotState {
    Eigen::Matrix<double, 7, 1> q;
    Eigen::Matrix<double, 7, 1> qp;
    Eigen::Matrix<double, 7, 1> qpp;
    bool valid = false;
};

struct SharedData {
    SharedRobotState robot_state;
    StopTimeResult stop_result;
    double t_stop_prev = 0.4;
};

// ============================================================================
// FUNZIONE SMOOTHSTEP
// ============================================================================
static double smoothstep(double edge0, double edge1, double x) {
    if (std::abs(edge1 - edge0) < 1e-12) return (x <= edge0) ? 1.0 : 0.0;
    double t = (x - edge0) / (edge1 - edge0);
    t = std::max(0.0, std::min(t, 1.0));
    return t * t * (3.0 - 2.0 * t);
}

// ============================================================================
// FUNZIONE PRINCIPALE
// ============================================================================

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname>\n";
        return -1;
    }

    pinocchio::Model pin_model;
    pinocchio::urdf::buildModel("panda-gazebo.urdf", pin_model);

    std::vector<Eigen::Vector3d> waypoints = {
        Eigen::Vector3d(0.3, 0.5, 0.5), Eigen::Vector3d(0.3, 0.5, 0.2),
        Eigen::Vector3d(0.5, 0.0, 0.5), Eigen::Vector3d(0.5, 0.0, 0.2),
        Eigen::Vector3d(0.3, -0.5, 0.5), Eigen::Vector3d(0.3, -0.5, 0.2),
        Eigen::Vector3d(0.5, 0.0, 0.4)
    };

    std::vector<int> pattern = {1, 3, 4, 3, 1, 2};
    std::vector<int> fullSequence;
    for(int k=0; k < 2; ++k) {
        for(int idx : pattern) fullSequence.push_back(idx);
    }

    const double segmentDuration = 3.0;
    int currentSequenceIndex = 0;
    double segmentTime = 0.0;

    DynamicSafetyParams safetyParams;
    safetyParams.T_reaction = 0.005;
    safetyParams.v_human_max = 1.6;
    safetyParams.csi = 0.0;
    safetyParams.rv = {0.085, 0.085, 0.06, 0.065};
    safetyParams.tau_max << 87, 87, 87, 87, 12, 12, 12;
    safetyParams.tau_rate_max << 1000, 1000, 1000, 1000, 1000, 1000, 1000;
    safetyParams.w0 = 1e6;
    safetyParams.w1 = 1e5;

    std::vector<franka::RobotState> log_data;
    log_data.reserve(200000);

    Eigen::Matrix<double, 4, 4> nextTrajPosition;
    nextTrajPosition.setIdentity();

    std::vector<Eigen::MatrixXd> coeffsX(7), coeffsY(7), coeffsZ(7);

    enum State { RUNNING, STOPPING, PAUSED, RECOVERING, FINISHED };
    State currentState = RUNNING;

    double stopTimer = 0.0;
    double recoveryTimer = 0.0;
    double recoveryDuration = 1.0;
    double frozenTime = 0.0;
    std::vector<Eigen::MatrixXd> recCoeffs(7);

    double globalTime = 0.0;
    bool firstRun = true;
    Eigen::Vector3d startPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d endPos = Eigen::Vector3d::Zero();
    Eigen::Vector3d lastDesiredPos = Eigen::Vector3d::Zero();

    // VARIABILI MULTITHREADING
    SharedData shared_data;
    std::mutex data_mutex;
    std::atomic<bool> program_running{true};

    // Inizializza StopTimeResult per sicurezza prima che il thread parta
    shared_data.stop_result.t_stop = 0.4;
    shared_data.stop_result.feasible = false;

    SkeletonZmqSubscriber skelSub("ipc:///tmp/skeleton.ipc");
    skelSub.start();
    auto skel_t0 = std::chrono::steady_clock::now();

    try {
        franka::Robot robot(argv[1]);
        setDefaultBehavior(robot);
        robot.setLoad(0.0, {0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
        franka::Model model = robot.loadModel();
        franka::RobotState initial_state = robot.readOnce();

        // --------------------------------------------------------------------
        // THREAD OTTIMIZZATORE CASADI
        // --------------------------------------------------------------------
        std::thread optimizer_thread([&]() {
            while (program_running) {
                SharedRobotState local_state;
                double local_t_prev;
                
                // 1. Lettura dello stato corrente (blocco breve)
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    local_state = shared_data.robot_state;
                    local_t_prev = shared_data.t_stop_prev;
                }

                // 2. Esecuzione CasADi (pesante, fuori dal mutex)
                if (local_state.valid) {
                    StopTimeResult result = optimize_stop_time_casadi_hybrid(
                        local_state.q, local_state.qp, local_state.qpp,
                        local_t_prev, safetyParams, model, initial_state, pin_model
                    );

                    // 3. Scrittura del risultato (blocco breve)
                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        shared_data.stop_result = result;
                        shared_data.t_stop_prev = result.t_stop;
                    }
                }

                // Frequenza di ottimizzazione: ~50 Hz (20 ms) per evitare sovraccarico CPU
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        });

        // Spostamento iniziale al punto E
        Eigen::Vector3d startPos_desired = waypoints[2];
        Eigen::Matrix3d startRot_desired;
        startRot_desired << 1, 0, 0, 0, -1, 0, 0, 0, -1;
        Eigen::Quaterniond startOri_desired(startRot_desired);

        std::cout << "Spostamento al punto E...\n";
        Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE_d.data()));
        Eigen::Vector3d p0 = initial_transform.translation();
        Eigen::Quaterniond q0(initial_transform.rotation());

        double dist = (startPos_desired - p0).norm();
        double angle_dist = q0.angularDistance(startOri_desired);

        if (dist > 0.01 || angle_dist > 0.01) {
            double duration = std::max(2.0, std::max(dist / 0.1, angle_dist / 0.5));
            
            // Calcolo coefficienti per traiettoria polinomiale di quinto grado
            std::vector<Eigen::MatrixXd> cx(7), cy(7), cz(7);
            for (int j = 0; j < 1; ++j) {
                cx[j] = compute_traj_poly5_coeffs(p0.x(), 0.0, 0.0, startPos_desired.x(), 0.0, 0.0, duration);
                cy[j] = compute_traj_poly5_coeffs(p0.y(), 0.0, 0.0, startPos_desired.y(), 0.0, 0.0, duration);
                cz[j] = compute_traj_poly5_coeffs(p0.z(), 0.0, 0.0, startPos_desired.z(), 0.0, 0.0, duration);
            }

            double time = 0.0;
            robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::CartesianPose {
                time += period.toSec();
                if (time > duration) {
                    Eigen::Affine3d final_transform(startOri_desired);
                    final_transform.translation() = startPos_desired;
                    std::array<double, 16> final_pose;
                    Eigen::Map<Eigen::Matrix4d>(final_pose.data()) = final_transform.matrix();
                    return franka::MotionFinished(final_pose);
                }

                double t2 = time*time, t3 = t2*time, t4 = t3*time, t5 = t4*time;
                Eigen::Vector3d pd;
                pd.x() = cx[0](0) + cx[0](1)*time + cx[0](2)*t2 + cx[0](3)*t3 + cx[0](4)*t4 + cx[0](5)*t5;
                pd.y() = cy[0](0) + cy[0](1)*time + cy[0](2)*t2 + cy[0](3)*t3 + cy[0](4)*t4 + cy[0](5)*t5;
                pd.z() = cz[0](0) + cz[0](1)*time + cz[0](2)*t2 + cz[0](3)*t3 + cz[0](4)*t4 + cz[0](5)*t5;

                double alpha = smoothstep(0.0, duration, time);
                Eigen::Quaterniond q_curr = q0.slerp(alpha, startOri_desired);

                Eigen::Affine3d new_transform(q_curr);
                new_transform.translation() = pd;
                std::array<double, 16> new_pose;
                Eigen::Map<Eigen::Matrix4d>(new_pose.data()) = new_transform.matrix();
                return new_pose;
            });
        }
        
        std::cout << "Avvio sequenza pick & place con capsule dinamiche (Pinocchio + Multithreading)...\n";

        // --------------------------------------------------------------------
        // LOOP DI CONTROLLO FRANKA (1 kHz HARD REAL-TIME)
        // --------------------------------------------------------------------
        robot.control([&](const franka::RobotState& state, franka::Duration duration) -> franka::CartesianPose {
            if (log_data.size() < log_data.capacity()) log_data.push_back(state);
            double dt = duration.toSec();
            globalTime += dt;

            if (firstRun) {
                Eigen::Map<const Eigen::Matrix4d> O_T_EE_start(state.O_T_EE_d.data());
                startPos = O_T_EE_start.block<3, 1>(0, 3);
                if (fullSequence.empty()) currentState = FINISHED;
                else {
                    endPos = waypoints[fullSequence[0]];
                    for (int j = 0; j < 1; ++j) {
                        coeffsX[j] = compute_traj_poly5_coeffs(startPos[0], 0.0, 0.0, endPos[0], 0.0, 0.0, segmentDuration);
                        coeffsY[j] = compute_traj_poly5_coeffs(startPos[1], 0.0, 0.0, endPos[1], 0.0, 0.0, segmentDuration);
                        coeffsZ[j] = compute_traj_poly5_coeffs(startPos[2], 0.0, 0.0, endPos[2], 0.0, 0.0, segmentDuration);
                    }
                }
                Eigen::Map<const Eigen::Matrix4d> current_pose(state.O_T_EE_d.data());
                nextTrajPosition = current_pose;
                lastDesiredPos = startPos;
                segmentTime = 0.0;
                currentSequenceIndex = 0;
                firstRun = false;
            }

            Eigen::Matrix<double, 7, 1> q_current, qp_current, qpp_current;
            for (int i = 0; i < 7; ++i) {
                q_current(i) = state.q[i];
                qp_current(i) = state.dq[i];
                qpp_current(i) = state.ddq_d[i];
            }

            // COMUNICAZIONE MULTITHREAD: Aggiornamento stato e lettura ottimizzazione
            StopTimeResult local_stop_result;
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                shared_data.robot_state.q = q_current;
                shared_data.robot_state.qp = qp_current;
                shared_data.robot_state.qpp = qpp_current;
                shared_data.robot_state.valid = true;
                local_stop_result = shared_data.stop_result; // Copia locale sicura
            }
            double t_stop_current = local_stop_result.t_stop;

            // Calcolo Frame e Capsule...
            std::array<Eigen::Vector3d, 8> frames;
            frames[0] = Eigen::Vector3d::Zero();
            auto get_pos = [&](franka::Frame f) {
                std::array<double, 16> p = model.pose(f, state);
                return Eigen::Vector3d(p[12], p[13], p[14]);
            };
            frames[1] = get_pos(franka::Frame::kJoint1); frames[2] = get_pos(franka::Frame::kJoint2);
            frames[3] = get_pos(franka::Frame::kJoint3); frames[4] = get_pos(franka::Frame::kJoint4);
            frames[5] = get_pos(franka::Frame::kJoint5); frames[6] = get_pos(franka::Frame::kJoint6);
            frames[7] = get_pos(franka::Frame::kFlange);

            std::array<CapsuleGeo, 4> robotCapsules;
            robotCapsules[0] = {frames[1], frames[0], safetyParams.rv[0]};
            robotCapsules[1] = {frames[3], frames[2], safetyParams.rv[1]};
            robotCapsules[2] = {frames[5], frames[4], safetyParams.rv[2]};
            robotCapsules[3] = {frames[7], frames[6], safetyParams.rv[3]};

            SkeletonCapsuleBuffer skel;
            skelSub.readLatest(skel);

            const uint64_t now_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - skel_t0).count();
            const double max_age_s = 0.10;
            const bool skeleton_valid = (skel.n_caps > 0) && (skel.rx_time_ns > 0) &&
                ((now_ns >= skel.rx_time_ns) ? ((now_ns - skel.rx_time_ns) * 1e-9 < max_age_s) : true);

            double human_radius = skeleton_valid ? skel.caps[0].radius : 0.0;
            double globalMinDistance = 1e9; 
            std::array<double, 4> dynamic_radii = {0.0, 0.0, 0.0, 0.0};

            if (skeleton_valid && currentState == RUNNING) {
                std::array<DistanceResult, 4> capsule_distances;
                for(int i=0; i<4; ++i) capsule_distances[i].distance = 1e9;
                
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < (int)skel.n_caps; ++j) {
                        DistanceResult res = distance_to_segment(
                            robotCapsules[i].p_start, robotCapsules[i].p_end, skel.caps[j].p_start, skel.caps[j].p_end
                        );
                        if (res.distance < capsule_distances[i].distance) capsule_distances[i] = res;
                    }
                    if(capsule_distances[i].distance < globalMinDistance) globalMinDistance = capsule_distances[i].distance;
                }

                // Utilizziamo il local_stop_result prelevato in modo asincrono
                dynamic_radii = compute_dynamic_capsule_radii(
                    local_stop_result, q_current, qp_current, capsule_distances, safetyParams, model, state
                );

                for (int i = 0; i < 4; ++i) {
                    double separation_distance = capsule_distances[i].distance - human_radius;
                    if (separation_distance <= dynamic_radii[i]) {
                        std::cout << "!!! STOP su Capsula " << i << " !!!\n";
                        currentState = STOPPING;
                        stopTimer = 0.0;
                        frozenTime = segmentTime;
                        break; 
                    }
                }
            } else if (skeleton_valid) {
                 for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < (int)skel.n_caps; ++j) {
                        DistanceResult res = distance_to_segment(
                            robotCapsules[i].p_start, robotCapsules[i].p_end, skel.caps[j].p_start, skel.caps[j].p_end
                        );
                        if(res.distance < globalMinDistance) globalMinDistance = res.distance;
                    }
                 }
            }

            // GESTIONE STATI
            if (currentState == STOPPING) {
                stopTimer += dt;
                
                double progress = stopTimer / t_stop_current;
                if (progress > 1.0) progress = 1.0;

                // Usa local_stop_result per l'interpolazione della frenata
                size_t n_samples = local_stop_result.q_traj.size();
                if (n_samples > 1) {
                    double sample_idx = progress * (n_samples - 1);
                    size_t idx_low = std::floor(sample_idx);
                    size_t idx_high = std::ceil(sample_idx);
                    double alpha = sample_idx - idx_low;
                    
                    Eigen::VectorXd q_safe_now = (1.0 - alpha) * local_stop_result.q_traj[idx_low] + 
                                                 alpha * local_stop_result.q_traj[idx_high];

                    franka::RobotState temp_state = state;
                    Eigen::Map<Eigen::Matrix<double, 7, 1>>(&temp_state.q[0]) = q_safe_now;
                    
                    std::array<double, 16> safe_pose_array = model.pose(franka::Frame::kEndEffector, temp_state);
                    Eigen::Map<Eigen::Matrix4d> safe_pose_mat(safe_pose_array.data());
                    lastDesiredPos = safe_pose_mat.block<3,1>(0,3);

                    if (stopTimer >= t_stop_current) {
                        currentState = PAUSED;
                        std::cout << "Robot fermo.\n";
                    }
                    return franka::CartesianPose(safe_pose_array);
                }
            } else if (currentState == PAUSED) {
                if (globalMinDistance > 0.3) {
                    currentState = RECOVERING;
                    recoveryTimer = 0.0;

                    double t2 = frozenTime*frozenTime, t3 = t2*frozenTime;
                    double t4 = t3*frozenTime, t5 = t4*frozenTime;
                    
                    Eigen::Vector3d target_pos;
                    target_pos.x() = coeffsX[0](0) + coeffsX[0](1)*frozenTime + coeffsX[0](2)*t2 + 
                                     coeffsX[0](3)*t3 + coeffsX[0](4)*t4 + coeffsX[0](5)*t5;
                    target_pos.y() = coeffsY[0](0) + coeffsY[0](1)*frozenTime + coeffsY[0](2)*t2 + 
                                     coeffsY[0](3)*t3 + coeffsY[0](4)*t4 + coeffsY[0](5)*t5;
                    target_pos.z() = coeffsZ[0](0) + coeffsZ[0](1)*frozenTime + coeffsZ[0](2)*t2 + 
                                     coeffsZ[0](3)*t3 + coeffsZ[0](4)*t4 + coeffsZ[0](5)*t5;
                    // Calcolo dinamico della durata del recovery basata sulla distanza minima raggiunta durante lo stop
                    // 0.25 m/s è la velocità massima del robot in ambito safety (ISO 15066)
                    // v_{peak} = 1.875 * v_{mean} quando si considerano curve quintiche
                    double dist_rec = (target_pos - lastDesiredPos).norm();
                    recoveryDuration = std::max(0.5, (1.875 * dist_rec) / 0.25);

                    for (int j = 0; j < 3; ++j) {
                        recCoeffs[j] = compute_traj_poly5_coeffs(
                            lastDesiredPos(j % 3), 0.0, 0.0, 
                            target_pos(j % 3), 0.0, 0.0, 
                            recoveryDuration
                        );
                    }

                    std::cout << "Recovery (T=" << recoveryDuration << "s)\n";
                }

            } else if (currentState == RECOVERING) {
                recoveryTimer += dt;

                double t2 = recoveryTimer*recoveryTimer, t3 = t2*recoveryTimer;
                double t4 = t3*recoveryTimer, t5 = t4*recoveryTimer;
                
                Eigen::Vector3d rec_pos;
                rec_pos.x() = recCoeffs[0](0) + recCoeffs[0](1)*recoveryTimer + recCoeffs[0](2)*t2 + 
                             recCoeffs[0](3)*t3 + recCoeffs[0](4)*t4 + recCoeffs[0](5)*t5;
                rec_pos.y() = recCoeffs[1](0) + recCoeffs[1](1)*recoveryTimer + recCoeffs[1](2)*t2 + 
                             recCoeffs[1](3)*t3 + recCoeffs[1](4)*t4 + recCoeffs[1](5)*t5;
                rec_pos.z() = recCoeffs[2](0) + recCoeffs[2](1)*recoveryTimer + recCoeffs[2](2)*t2 + 
                             recCoeffs[2](3)*t3 + recCoeffs[2](4)*t4 + recCoeffs[2](5)*t5;

                Eigen::Map<Eigen::Matrix4d> desired_pose(nextTrajPosition.data());
                desired_pose.block<3, 1>(0, 3) = rec_pos;
                lastDesiredPos = rec_pos;

                if (recoveryTimer >= recoveryDuration) {
                    currentState = RUNNING;
                    segmentTime = frozenTime;
                    std::cout << "Recovery completato.\n";
                }

            } else if (currentState == RUNNING) {
                segmentTime += dt;

                if (segmentTime <= segmentDuration) {
                    double t2 = segmentTime*segmentTime, t3 = t2*segmentTime;
                    double t4 = t3*segmentTime, t5 = t4*segmentTime;
                    
                    Eigen::Vector3d traj_pos;
                    traj_pos.x() = coeffsX[0](0) + coeffsX[0](1)*segmentTime + coeffsX[0](2)*t2 + 
                                   coeffsX[0](3)*t3 + coeffsX[0](4)*t4 + coeffsX[0](5)*t5;
                    traj_pos.y() = coeffsY[0](0) + coeffsY[0](1)*segmentTime + coeffsY[0](2)*t2 + 
                                   coeffsY[0](3)*t3 + coeffsY[0](4)*t4 + coeffsY[0](5)*t5;
                    traj_pos.z() = coeffsZ[0](0) + coeffsZ[0](1)*segmentTime + coeffsZ[0](2)*t2 + 
                                   coeffsZ[0](3)*t3 + coeffsZ[0](4)*t4 + coeffsZ[0](5)*t5;

                    Eigen::Map<Eigen::Matrix4d> desired_pose(nextTrajPosition.data());
                    desired_pose.block<3, 1>(0, 3) = traj_pos;
                    lastDesiredPos = traj_pos;

                } else {
                    currentSequenceIndex++;
                    if (currentSequenceIndex >= fullSequence.size()) {
                        currentState = FINISHED;
                        std::cout << "Sequenza completata!\n";
                    } else {
                        startPos = endPos;
                        endPos = waypoints[fullSequence[currentSequenceIndex]];
                        std::cout << "Segmento " << currentSequenceIndex + 1 << "/" << fullSequence.size() << "\n";

                        for (int j = 0; j < 1; ++j) {
                            coeffsX[j] = compute_traj_poly5_coeffs(startPos[0], 0.0, 0.0, endPos[0], 0.0, 0.0, segmentDuration);
                            coeffsY[j] = compute_traj_poly5_coeffs(startPos[1], 0.0, 0.0, endPos[1], 0.0, 0.0, segmentDuration);
                            coeffsZ[j] = compute_traj_poly5_coeffs(startPos[2], 0.0, 0.0, endPos[2], 0.0, 0.0, segmentDuration);
                        }
                        segmentTime = 0.0;
                    }
                }
            }

            if ((int)(globalTime * 1000) % 1000 == 0 && currentState == RUNNING) {
                std::cout << "Seq: " << currentSequenceIndex << "/" << fullSequence.size()
                          << " | t_seg: " << std::fixed << std::setprecision(2) << segmentTime
                          << " | t_stop: " << t_stop_current
                          << " | Min_dist: " << std::fixed << std::setprecision(3) << globalMinDistance
                          << " | Dyn_radii: [" << std::fixed << std::setprecision(3) 
                          << dynamic_radii[0] << ", " << dynamic_radii[1] << ", " 
                          << dynamic_radii[2] << ", " << dynamic_radii[3] << "]"
                          << "\n";
            }

            std::array<double, 16> desiredPose;
            Eigen::Map<Eigen::Matrix4d>(desiredPose.data()) = nextTrajPosition;

            if (currentState == FINISHED) {
                program_running = false; // Segnala al thread CasADi di terminare
                return franka::MotionFinished(desiredPose);
            }

            return desiredPose;
        });

        skelSub.stop();
        
        // Attende in modo sicuro la chiusura del thread
        if (optimizer_thread.joinable()) {
            optimizer_thread.join();
        }

    } catch (const franka::Exception& ex) {
        program_running = false; // Arresta il thread in caso di eccezione
        std::cerr << "Eccezione Franka: " << ex.what() << "\n";
        save_log_to_csv(log_data, "log_error_dynamic.csv");
        return -1;
    } catch (const std::exception& e) {
        program_running = false; // Arresta il thread in caso di eccezione
        std::cerr << "Eccezione: " << e.what() << "\n";
        save_log_to_csv(log_data, "log_error_dynamic.csv");
        return -1;
    }

    save_log_to_csv(log_data, "log_dynamic_success.csv");
    std::cout << "Programma completato.\n";
    return 0;
}
