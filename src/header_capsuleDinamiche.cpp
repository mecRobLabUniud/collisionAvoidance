#include "header_capsuleDinamiche.h"
#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/algorithm/rnea.hpp>

using namespace casadi;

// ============================================================================
// OTTIMIZZAZIONE TEMPO DI STOP CON CASADI E PINOCCHIO
// ============================================================================
// Questa funzione ottimizza il tempo necessario per fermare il robot in sicurezza
// utilizzando CasADi per la formulazione simbolica e Pinocchio per la dinamica.
// Restituisce una traiettoria ottimizzata e il tempo di stop calcolato.

StopTimeResult optimize_stop_time_casadi_hybrid(
    const Eigen::Matrix<double, 7, 1>& q_current,     // Posizioni correnti dei giunti
    const Eigen::Matrix<double, 7, 1>& qp_current,    // Velocità correnti dei giunti
    const Eigen::Matrix<double, 7, 1>& qpp_current,   // Accelerazioni correnti dei giunti
    double t_stop_prev,                               // Tempo di stop precedente per smoothing
    const DynamicSafetyParams& params,                // Parametri di sicurezza
    const franka::Model& model_franka,                // Modello Franka
    const franka::RobotState& state,                  // Stato corrente del robot
    const pinocchio::Model& pin_model                 // Modello Pinocchio
) {
    StopTimeResult result;
    result.t_stop = 0.4;  // Valore di default
    result.feasible = false;
    
    try {
        static Function solver;  // Solver CasADi (inizializzato una volta)
        static bool solver_ready = false;

        if (!solver_ready) {
            // ----------------------------------------------------------------
            // 1. DEFINIZIONE VARIABILI SIMBOLICHE (CasADi SX)
            // ----------------------------------------------------------------
            // Variabili simboliche: x = tempo di stop, p = parametri (22 elementi)
            // 'x' è l'unica variabile di ottimizzazione: il tempo T necessario per fermarsi.
            SX x = SX::sym("x"); // t_stop
            // 'p' contiene lo stato iniziale (q, dq, ddq) e il t_stop precedente (per warm start/smoothing)
            SX p = SX::sym("p", 22); 
            
            SX t_prev_sym = p(21);  // Tempo di stop precedente

            // Funzione di costo: minimizza tempo di stop con penalità per variazioni brusche
            SX cost = params.w0 * x + params.w1 * sqrt(pow(x - t_prev_sym, 2) + 1e-6);
            
            int n = 10;  // Numero di punti per discretizzazione traiettoria
            double toll = 0.9;  // Tolleranza per limiti
            
            // Limiti di velocità, accelerazione, jerk per giunti
            std::vector<double> qp_lim = {2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100};
            std::vector<double> qpp_lim = {15, 7.5, 10, 12.5, 15, 20, 20};
            std::vector<double> qppp_lim = {7500, 3750, 5000, 6250, 7500, 10000, 10000};
            
            // Limiti di posizione per giunti
            std::vector<double> q_min = {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
            std::vector<double> q_max = {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
            
            // ----------------------------------------------------------------
            // 2. COSTRUZIONE TRAIETTORIA SIMBOLICA (Polinomio grado 5)
            // ----------------------------------------------------------------
            // Costruzione della traiettoria simbolica polinomiale per il movimento di stop
            std::vector<SX> Q(n + 1), Qp(n + 1), Qpp(n + 1), Qppp(n + 1);
            
            for (int k = 0; k <= n; ++k) {
                Q[k] = SX::zeros(7, 1);
                Qp[k] = SX::zeros(7, 1);
                Qpp[k] = SX::zeros(7, 1);
                Qppp[k] = SX::zeros(7, 1);
            }
            
            // Per ogni giunto, calcoliamo i coefficienti del polinomio in funzione di 'x' (tempo T)
            // e dello stato iniziale 'p'.
            for (int j = 0; j < 7; ++j) {
                SX qi = p(j);
                SX qpi = p(j + 7);
                SX qppi = p(j + 14);
                
                SX a0 = qi;
                SX a1 = qpi;
                SX a2 = qppi / 2.0;
                
                // Il sistema lineare per trovare a3, a4, a5 dipende simbolicamente da T (x)
                SX T = x;
                // Condizioni finali: posizione libera, ma vel=0, acc=0 al tempo T
                SX b0 = 0 - qpi*T - 0.5*qppi*pow(T, 2);
                SX b1 = 0 - qpi - qppi*T;
                SX b2 = 0 - qppi;
                
                // Matrice del sistema lineare simbolico
                SX V = SX::vertcat({
                    SX::horzcat({pow(T,3), pow(T,4), pow(T,5)}),
                    SX::horzcat({3*pow(T,2), 4*pow(T,3), 5*pow(T,4)}),
                    SX::horzcat({6*T, 12*pow(T,2), 20*pow(T,3)})
                });
                
                SX knowns = SX::vertcat({b0, b1, b2});
                // Risoluzione simbolica: a_345 contiene espressioni che dipendono da x
                // Risolve il sistema lineare V * [a3; a4; a5] = knowns per trovare i coefficienti del polinomio in funzione di T (x)
                // Nota che CasADi costruisce un grafo computazionale che tiene traccia di tutte le operazioni, permettendo di calcolare derivate rispetto a 'x' in modo efficiente. Pertanto a_345 sarà una funzione simbolica di 'x' che rappresenta i coefficienti del polinomio in funzione del tempo di stop che stiamo ottimizzando (è un vettore di formule simboliche).
                SX a_345 = solve(V, knowns);
                
                SX a3 = a_345(0);
                SX a4 = a_345(1);
                SX a5 = a_345(2);
                
                // Campionamento della traiettoria simbolica in n punti
                // NOTA: Qui siamo ancora nel dominio SIMBOLICO. Non stiamo calcolando valori numerici.
                // 't' è un'espressione che dipende dalla variabile di ottimizzazione 'T' (x).
                // Q[k] diventa una formula: "posizione al k-esimo istante percentuale della durata T".
                for (int k = 0; k <= n; ++k) {
                    SX t = (static_cast<double>(k) / n) * T;
                    SX t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
                    
                    Q[k](j) = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5;
                    Qp[k](j) = a1 + 2*a2*t + 3*a3*t2 + 4*a4*t3 + 5*a5*t4;
                    Qpp[k](j) = 2*a2 + 6*a3*t + 12*a4*t2 + 20*a5*t3;
                    Qppp[k](j) = 6*a3 + 24*a4*t + 60*a5*t2;
                }
            }
            
            SXVector g;
            
            // ----------------------------------------------------------------
            // 3. VINCOLI CINEMATICI (Posizione, Velocità, Accelerazione, Jerk)
            // ----------------------------------------------------------------
            for (int k = 0; k <= n; ++k) {
                for (int j = 0; j < 7; ++j) {
                    g.push_back(q_min[j] * toll - Q[k](j));
                    g.push_back(Q[k](j) - q_max[j] * toll);
                    g.push_back(fabs(Qp[k](j)) - qp_lim[j] * toll);
                    g.push_back(fabs(Qpp[k](j)) - qpp_lim[j] * toll);
                    g.push_back(fabs(Qppp[k](j)) - qppp_lim[j] * toll);
                }
            }
            
            // ----------------------------------------------------------------
            // 4. VINCOLI DINAMICI (Pinocchio + CasADi)
            // ----------------------------------------------------------------
            // Qui avviene l'integrazione chiave: usiamo i template di Pinocchio
            // istanziati con il tipo scalare simbolico di CasADi (SX).
            typedef pinocchio::ModelTpl<SX> ModelAD;
            typedef pinocchio::DataTpl<SX> DataAD;
            
            // Cast del modello numerico (double) a modello simbolico (SX).
            // Le costanti fisiche (masse, inerzie) diventano nodi costanti nel grafo CasADi.
            ModelAD model_ad = pin_model.cast<SX>();
            DataAD data_ad(model_ad);

            Eigen::Matrix<SX, 7, 1> tau_prev; // Per memorizzare la coppia allo step precedente

            for (int k = 0; k <= n; ++k) {
                Eigen::Matrix<SX, 7, 1> q_ad, qp_ad, qpp_ad;
                for(int i=0; i<7; ++i) {
                    q_ad(i) = Q[k](i); qp_ad(i) = Qp[k](i); qpp_ad(i) = Qpp[k](i);
                }
                
                // RNEA (Recursive Newton-Euler Algorithm) Simbolico.
                // Calcola tau = M(q)*qpp + C(q,qp)*qp + g(q) costruendo il grafo delle operazioni.
                // CasADi traccerà automaticamente le derivate di queste operazioni rispetto a 'x'.
                pinocchio::rnea(model_ad, data_ad, q_ad, qp_ad, qpp_ad);
                
                Eigen::Matrix<SX, 7, 1> tau_curr;
                for(int i=0; i<7; ++i) tau_curr(i) = data_ad.tau(i);
                
                for (int j = 0; j < 7; ++j) {
                    SX tau_j = data_ad.tau(j);
                    g.push_back(tau_j - params.tau_max(j) * toll);
                    g.push_back(-params.tau_max(j) * toll - tau_j);
                }
                
                // Vincoli su derivata coppia (tau_p)
                if (k > 0) {
                    SX dt = x / double(n); // dt = T_stop / n_samples
                    for (int j = 0; j < 7; ++j) {
                        SX tau_dot = (tau_curr(j) - tau_prev(j)) / dt;
                        g.push_back(tau_dot - params.tau_rate_max(j) * toll);
                        g.push_back(-params.tau_rate_max(j) * toll - tau_dot);
                    }
                }
                tau_prev = tau_curr;
            }

            // ----------------------------------------------------------------
            // 5. CREAZIONE DEL SOLVER NLP
            // ----------------------------------------------------------------
            SXDict nlp;
            nlp["x"] = x;           // Variabile di ottimizzazione (t_stop)
            nlp["p"] = p;           // Parametri (stato iniziale q, dq, ddq + t_prev)
            nlp["f"] = cost;        // Funzione obiettivo da minimizzare
            nlp["g"] = vertcat(g);  // Vettore colonna di tutti i vincoli (cinematici + dinamici)
            
            Dict opts;
            // Opzioni per esecuzione Real-Time (silenzioso e veloce)
            opts["ipopt.print_level"] = 0;       // Disabilita output su console (fondamentale per loop veloci)
            opts["print_time"] = 0;              // Non stampare statistiche temporali
            opts["ipopt.sb"] = "yes";            // Nasconde banner iniziale IPOPT
            
            // Parametri critici per performance Real-Time
            opts["ipopt.max_iter"] = 10;         // Limita iterazioni per garantire tempo di calcolo deterministico (<1ms)
            opts["ipopt.tol"] = 1e-4;            // Tolleranza convergenza desiderata
            opts["ipopt.acceptable_tol"] = 1e-3; // Tolleranza accettabile se max_iter raggiunto
            opts["ipopt.warm_start_init_point"] = "yes"; // Usa la soluzione precedente come punto di partenza (velocizza convergenza)
            
            solver = nlpsol("solver", "ipopt", nlp, opts); // Compilazione del solver
            solver_ready = true;
        }
        
        std::map<std::string, DM> arg, res;
        arg["x0"] = t_stop_prev;
        arg["lbx"] = 0.01;
        arg["ubx"] = 0.4;
        arg["lbg"] = -inf;
        arg["ubg"] = 0;

        std::vector<double> p_val(22);
        for(int i=0; i<7; ++i) {
            p_val[i] = q_current(i);
            p_val[i+7] = qp_current(i);
            p_val[i+14] = qpp_current(i);
        }
        p_val[21] = t_stop_prev;
        
        arg["p"] = p_val;
        
        res = solver(arg);
        DM x_opt = res.at("x");
        double t_opt = static_cast<double>(x_opt);
        
        generate_quintic_trajectory(
            q_current, qp_current, qpp_current,
            q_current, Eigen::Matrix<double, 7, 1>::Zero(), Eigen::Matrix<double, 7, 1>::Zero(),
            t_opt, 10,
            result.q_traj, result.qp_traj, result.qpp_traj
        );
        
        // Verifica finale numerica con libfranka come sicurezza aggiuntiva
        bool torque_ok = check_torque_limits_numeric(result.q_traj, result.qp_traj, result.qpp_traj, params, model_franka, state);
        
        if (torque_ok) {
            result.t_stop = t_opt;
            result.feasible = true;
        } else {
            result.t_stop = 0.4;
            result.feasible = false;
            generate_quintic_trajectory(
                q_current, qp_current, qpp_current,
                q_current, Eigen::Matrix<double, 7, 1>::Zero(), Eigen::Matrix<double, 7, 1>::Zero(),
                0.4, 10,
                result.q_traj, result.qp_traj, result.qpp_traj
            );
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[CasADi] Errore: " << e.what() << std::endl;
        result.t_stop = 0.4;
        result.feasible = false;
        generate_quintic_trajectory(
            q_current, qp_current, qpp_current,
            q_current, Eigen::Matrix<double, 7, 1>::Zero(), Eigen::Matrix<double, 7, 1>::Zero(),
            0.4, 10,
            result.q_traj, result.qp_traj, result.qpp_traj
        );
    }
    
    return result;
}

// ============================================================================
// VERIFICA VINCOLI COPPIA NUMERICA
// ============================================================================

bool check_torque_limits_numeric(
    const std::vector<Eigen::VectorXd>& q_traj,
    const std::vector<Eigen::VectorXd>& qp_traj,
    const std::vector<Eigen::VectorXd>& qpp_traj,
    const DynamicSafetyParams& params,
    const franka::Model& model,
    const franka::RobotState& state
) {
    franka::RobotState temp_state = state;
    
    for (size_t k = 0; k < q_traj.size(); ++k) {
        for (int j = 0; j < 7; ++j) {
            temp_state.q[j] = q_traj[k](j);
            temp_state.dq[j] = qp_traj[k](j);
            temp_state.ddq_d[j] = qpp_traj[k](j);
        }
        
        std::array<double, 49> mass_array = model.mass(temp_state);
        std::array<double, 7> coriolis_array = model.coriolis(temp_state);
        std::array<double, 7> gravity_array = model.gravity(temp_state);
        
        Eigen::Map<const Eigen::Matrix<double, 7, 7>> M(mass_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> c(coriolis_array.data());
        Eigen::Map<const Eigen::Matrix<double, 7, 1>> g(gravity_array.data());
        
        Eigen::Matrix<double, 7, 1> tau = M * qpp_traj[k] + c + g;
        
        for (int j = 0; j < 7; ++j) {
            if (std::abs(tau(j)) > params.tau_max(j) * 0.95) {
                return false;
            }
        }
    }
    
    return true;
}

// ============================================================================
// CALCOLO RAGGI CAPSULE DINAMICHE
// ============================================================================

std::array<double, 4> compute_dynamic_capsule_radii(
    const StopTimeResult& stop_result,
    const Eigen::Matrix<double, 7, 1>& q_current,
    const Eigen::Matrix<double, 7, 1>& qp_current,
    const std::array<DistanceResult, 4>& capsule_distances,
    const DynamicSafetyParams& params,
    const franka::Model& model,
    const franka::RobotState& state
) {
    std::array<double, 4> radii;
    
    double Sh = params.v_human_max * (params.T_reaction + stop_result.t_stop);
    
    std::array<std::pair<int, int>, 4> capsule_joints = {{
        {1, 0}, {3, 2}, {5, 4}, {7, 6}
    }};
    
    for (int cap_idx = 0; cap_idx < 4; ++cap_idx) {
        Eigen::Vector3d C_h = capsule_distances[cap_idx].closest_point_2;
        Eigen::Vector3d C_r = capsule_distances[cap_idx].closest_point_1;
        
        Eigen::Vector3d C_dir = C_h - C_r;
        double C_norm = C_dir.norm();
        if (C_norm < 1e-6) C_norm = 1e-6;
        Eigen::Vector3d C_unit = C_dir / C_norm;
        
        int joint_a = capsule_joints[cap_idx].first;
        int joint_b = capsule_joints[cap_idx].second;
        
        Eigen::Vector3d p_a = get_joint_position(model, state, static_cast<franka::Frame>(joint_a));
        Eigen::Vector3d p_b = (joint_b == 0) ? Eigen::Vector3d::Zero() : 
            get_joint_position(model, state, static_cast<franka::Frame>(joint_b));
        
        Eigen::MatrixXd J = compute_geometric_jacobian(model, state, joint_a);
        Eigen::Matrix<double, 3, 1> v_linear = J.block<3, 7>(0, 0) * qp_current;
        Eigen::Matrix<double, 3, 1> omega = J.block<3, 7>(3, 0) * qp_current;
        
        Eigen::Vector3d dir_capsule = (p_a - p_b).normalized();
        Eigen::Vector3d v_a_e = v_linear + omega.cross(dir_capsule * params.rv[cap_idx]);
        Eigen::Vector3d v_b_e = v_linear - omega.cross(dir_capsule * params.rv[cap_idx]);
        
        double v_a_proj = v_a_e.dot(C_unit);
        double v_b_proj = v_b_e.dot(C_unit);
        double v_max_current = std::max(v_a_proj, v_b_proj);
        
        if (v_max_current < 0.0) v_max_current = 0.0;
        
        double Sr = v_max_current * params.T_reaction;
        
        double Ss = 0.0;
        if (stop_result.q_traj.size() > 1) {
            double dt = stop_result.t_stop / (stop_result.q_traj.size() - 1);
            franka::RobotState temp_state = state;
            double v_prev = v_max_current;
            
            for (size_t k = 0; k < stop_result.q_traj.size(); ++k) {
                for (int j = 0; j < 7; ++j) {
                    temp_state.q[j] = stop_result.q_traj[k](j);
                    temp_state.dq[j] = stop_result.qp_traj[k](j);
                }
                
                Eigen::Vector3d p_a_k = get_joint_position(model, temp_state, static_cast<franka::Frame>(joint_a));
                Eigen::Vector3d p_b_k = (joint_b == 0) ? Eigen::Vector3d::Zero() : 
                    get_joint_position(model, temp_state, static_cast<franka::Frame>(joint_b));
                
                Eigen::MatrixXd J_k = compute_geometric_jacobian(model, temp_state, joint_a);
                Eigen::Matrix<double, 3, 1> v_linear_k = J_k.block<3, 7>(0, 0) * stop_result.qp_traj[k];
                Eigen::Matrix<double, 3, 1> omega_k = J_k.block<3, 7>(3, 0) * stop_result.qp_traj[k];
                
                Eigen::Vector3d dir_k = (p_a_k - p_b_k).normalized();
                Eigen::Vector3d v_a_e_k = v_linear_k + omega_k.cross(dir_k * params.rv[cap_idx]);
                Eigen::Vector3d v_b_e_k = v_linear_k - omega_k.cross(dir_k * params.rv[cap_idx]);
                
                Eigen::Vector3d C_h_k = C_h - C_unit * (params.v_human_max * k * dt);
                Eigen::Vector3d C_dir_k = C_h_k - p_a_k;
                double C_norm_k = C_dir_k.norm();
                if (C_norm_k < 1e-6) C_norm_k = 1e-6;
                Eigen::Vector3d C_unit_k = C_dir_k / C_norm_k;
                
                double v_a_proj_k = v_a_e_k.dot(C_unit_k);
                double v_b_proj_k = v_b_e_k.dot(C_unit_k);
                double v_max_k = std::max(v_a_proj_k, v_b_proj_k);
                
                if (v_max_k < 0.0) v_max_k = 0.0;
                
                if (k > 0) {
                    Ss += 0.5 * (v_max_k + v_prev) * dt;
                }
                v_prev = v_max_k;
            }
        }
        
        radii[cap_idx] = params.rv[cap_idx] + Sh + Sr + Ss + params.csi;
    }
    
    return radii;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

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
) {
    q_out.clear();
    qp_out.clear();
    qpp_out.clear();
    
    std::vector<Eigen::MatrixXd> coeffs(7);
    for (int j = 0; j < 7; ++j) {
        coeffs[j] = compute_traj_poly5_coeffs(
            q_start(j), qp_start(j), qpp_start(j),
            q_end(j), qp_end(j), qpp_end(j),
            duration
        );
    }
    
    for (int k = 0; k < n_samples; ++k) {
        double t = (k * duration) / (n_samples - 1);
        double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
        
        Eigen::VectorXd q(7), qp(7), qpp(7);
        
        for (int j = 0; j < 7; ++j) {
            q(j) = coeffs[j](0) + coeffs[j](1)*t + coeffs[j](2)*t2 + 
                   coeffs[j](3)*t3 + coeffs[j](4)*t4 + coeffs[j](5)*t5;
            qp(j) = coeffs[j](1) + 2*coeffs[j](2)*t + 3*coeffs[j](3)*t2 + 
                    4*coeffs[j](4)*t3 + 5*coeffs[j](5)*t4;
            qpp(j) = 2*coeffs[j](2) + 6*coeffs[j](3)*t + 12*coeffs[j](4)*t2 + 
                     20*coeffs[j](5)*t3;
        }
        
        q_out.push_back(q);
        qp_out.push_back(qp);
        qpp_out.push_back(qpp);
    }
}

Eigen::MatrixXd compute_traj_poly5_coeffs(
    double qi, double qi_p, double qi_pp,
    double qf, double qf_p, double qf_pp,
    double duration
) {
    Eigen::Vector3d coeff_known;
    coeff_known(0) = qi;
    coeff_known(1) = qi_p;
    coeff_known(2) = qi_pp / 2.0;
    
    Eigen::Vector3d known_terms;
    known_terms(0) = qf - qi - qi_p * duration - 0.5 * qi_pp * std::pow(duration, 2);
    known_terms(1) = qf_p - qi_p - qi_pp * duration;
    known_terms(2) = qf_pp - qi_pp;
    
    Eigen::Matrix3d V;
    V << std::pow(duration, 3), std::pow(duration, 4), std::pow(duration, 5),
         3 * std::pow(duration, 2), 4 * std::pow(duration, 3), 5 * std::pow(duration, 4),
         6 * duration, 12 * std::pow(duration, 2), 20 * std::pow(duration, 3);
    
    Eigen::Vector3d coeff_unknown = V.colPivHouseholderQr().solve(known_terms);
    
    Eigen::MatrixXd all_coeffs(6, 1);
    all_coeffs << coeff_known, coeff_unknown;
    
    return all_coeffs;
}

Eigen::Vector3d get_joint_position(
    const franka::Model& model,
    const franka::RobotState& state,
    franka::Frame frame
) {
    std::array<double, 16> transform = model.pose(frame, state);
    return Eigen::Vector3d(transform[12], transform[13], transform[14]);
}

Eigen::MatrixXd compute_geometric_jacobian(
    const franka::Model& model,
    const franka::RobotState& state,
    int k
) {
    franka::Frame frame = static_cast<franka::Frame>(k);
    std::array<double, 42> jac_array = model.zeroJacobian(frame, state);
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> J(jac_array.data());
    return J;
}

Eigen::Vector3d compute_capsule_endpoint_velocity(
    const Eigen::Vector3d& p_joint,
    const Eigen::Vector3d& omega_joint,
    const Eigen::Vector3d& p_endpoint,
    const Eigen::Vector3d& p_other_end,
    double rv
) {
    Eigen::Vector3d capsule_dir = p_endpoint - p_other_end;
    double norm = capsule_dir.norm();
    if (norm < 1e-6) norm = 1e-6;
    Eigen::Vector3d capsule_unit = capsule_dir / norm;
    return omega_joint.cross(capsule_unit * rv);
}

DistanceResult distance_to_segment(
    Eigen::Vector3d P1, Eigen::Vector3d Q1,
    Eigen::Vector3d P2, Eigen::Vector3d Q2
) {
    auto clamp_val = [](double n) { return std::max(0.0, std::min(1.0, n)); };
    
    Eigen::Vector3d D1 = Q1 - P1;
    Eigen::Vector3d D2 = Q2 - P2;
    Eigen::Vector3d R = P1 - P2;
    
    double a = D1.dot(D1);
    double b = D1.dot(D2);
    double c = D1.dot(R);
    double e = D2.dot(D2);
    double f = D2.dot(R);
    
    if (a <= 1e-9 && e <= 1e-9) return { (P1 - P2).norm(), P1, P2 };
    if (a <= 1e-9) {
        double t = clamp_val(-f / std::max(e, 1e-9));
        return { (P1 - (P2 + t * D2)).norm(), P1, P2 + t * D2 };
    }
    if (e <= 1e-9) {
        double s = clamp_val(-c / std::max(a, 1e-9));
        return { ((P1 + s * D1) - P2).norm(), P1 + s * D1, P2 };
    }
    
    double d = a * e - b * b;
    if (std::abs(d) < 1e-9) d = 1e-9;
    
    double s = clamp_val((b * f - c * e) / d);
    double t = (b * s + f) / e;
    
    if (t < 0.0) { t = 0.0; s = clamp_val(-c / a); }
    else if (t > 1.0) { t = 1.0; s = clamp_val((b - c) / a); }
    
    Eigen::Vector3d C_h = P1 + D1 * s;
    Eigen::Vector3d C_r = P2 + D2 * t;
    
    return {(C_h - C_r).norm(), C_h, C_r};
}

void save_log_to_csv(
    const std::vector<franka::RobotState>& log_data,
    const std::string& filename
) {
    if (log_data.empty()) return;
    
    std::ofstream log_file(filename);
    if (!log_file.is_open()) {
        std::cerr << "ERRORE: Impossibile creare " << filename << std::endl;
        return;
    }
    
    log_file << "time,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7\n";
    
    double time = 0.0;
    for (const auto& state : log_data) {
        log_file << time;
        for(double val : state.q) log_file << "," << val;
        for(double val : state.dq) log_file << "," << val;
        log_file << "\n";
        time += 0.001;
    }
    
    std::cout << "Log salvato: " << filename << std::endl;
}