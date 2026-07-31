#include "collision_avoidance.hpp"
#include <cmath>
#include <optional>
#include <limits>

namespace collision_avoidance {

// Skeleton segment indices (pairs of joint indices)
const std::array<std::array<int, 2>, 10> skel_index = {{
    {{0, 1}},
    {{2, 3}},
    {{3, 4}},
    {{5, 6}},
    {{6, 7}},
    {{1, 8}},
    {{9, 10}},
    {{10, 11}},
    {{12, 13}},
    {{13, 14}}
}};

// Robot capsule radii (rv)
const std::array<double, 4> rv = {0.085, 0.085, 0.06, 0.065};

// Safety parameters
const double T_reaction = 0.05;      // [s]
const double csi = 0.0;
const double max_human_speed = 1.6;
const double max_robot_speed = 2.0;

// Robot DH parameters (7x4: a, alpha, d, theta_offset)
const Eigen::Matrix<double, 7, 4> DH_default = []() {
    Eigen::Matrix<double, 7, 4> DH;
    const double pi = M_PI;
    DH << 0.0,    -pi/2.0,  0.0,     0.333,
          0.0,    -pi/2.0,  pi,      0.0,
          0.088,   pi/2.0,  pi,      0.316,
          0.088,   pi/2.0,  pi,      0.0,
          0.0,     pi/2.0,  pi,      0.384,
          0.088,   pi/2.0,  0.0,     0.0,
          0.0,     0.0,     0.0,     0.2;
    return DH;
}();

// Robot to marker offset
const Eigen::Vector3d robot_to_marker_pos(0.313, 0.006, -0.02);

// Human skeleton segment safety radii
const std::array<double, 10> r_sw_h = {
    0.16, 0.05, 0.05, 0.06, 0.06,
    0.15, 0.10, 0.10, 0.08, 0.08
};

double clamp(double n) {
    if (n < 0.0) return 0.0;
    if (n > 1.0) return 1.0;
    return n;
}

double trapz(const Eigen::VectorXd& t, const std::vector<double>& y) {
    double area = 0.0;
    for (size_t i = 1; i < t.size(); ++i) {
        double dt = t(i) - t(i - 1);
        double avg = 0.5 * (y[i] + y[i - 1]);
        area += dt * avg;
    }
    return area;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> TrajPoly5(
    double qi, double qi_p, double qi_pp,
    double qf, double qf_p, double qf_pp,
    double traj_duration, double time_step)
{
    const int N = static_cast<int>(std::floor(traj_duration * 1000.0 / time_step)) + 1;
    Eigen::Matrix<double, 3, Eigen::Dynamic> traj(3, N);

    double T = traj_duration;

    // Coefficients of 5th order polynomial:
    // q(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5
    double c0 = qi;
    double c1 = qi_p;
    double c2 = qi_pp / 2.0;

    Eigen::Vector3d known_terms;
    known_terms(0) = qf - qi - qi_p * T - 0.5 * qi_pp * T * T;
    known_terms(1) = qf_p - qi_p - qi_pp * T;
    known_terms(2) = qf_pp - qi_pp;

    // Vandermonde matrix for [c3, c4, c5]
    Eigen::Matrix3d V;
    V << std::pow(T, 3), std::pow(T, 4), std::pow(T, 5),
         3 * std::pow(T, 2), 4 * std::pow(T, 3), 5 * std::pow(T, 4),
         6 * T, 12 * std::pow(T, 2), 20 * std::pow(T, 3);

    Eigen::Vector3d coeff_1 = V.colPivHouseholderQr().solve(known_terms);

    double c3 = coeff_1(0);
    double c4 = coeff_1(1);
    double c5 = coeff_1(2);

    for (int i = 0; i < N; ++i) {
        double t = (i == 0) ? 0.0 : (T * static_cast<double>(i) / static_cast<double>(N - 1));
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;

        double q   = c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t4 + c5 * t5;
        double qd  = c1 + 2 * c2 * t + 3 * c3 * t2 + 4 * c4 * t3 + 5 * c5 * t4;
        double qdd = 2 * c2 + 6 * c3 * t + 12 * c4 * t2 + 20 * c5 * t3;

        traj(0, i) = q;
        traj(1, i) = qd;
        traj(2, i) = qdd;
    }

    return traj;
}

Eigen::Vector3d compute_capsule_end_point(
    const Eigen::Matrix<double, 7, 4>& DH,
    const Eigen::Vector<double, 7>& q,
    int k)
{
    Eigen::Matrix4d T_k = Eigen::Matrix4d::Identity();
    const double pi = M_PI;

    for (int j = 0; j < k; ++j) {
        double a     = DH(j, 0);
        double alpha = DH(j, 1);
        double d     = DH(j, 2);
        double theta_offset = DH(j, 3);

        double theta = q(j) + theta_offset;

        double ct = std::cos(theta);
        double st = std::sin(theta);
        double ca = std::cos(alpha);
        double sa = std::sin(alpha);

        Eigen::Matrix4d T;
        T << ct, -st * ca,  st * sa,  a * ct,
             st,  ct * ca, -ct * sa,  a * st,
             0,   sa,       ca,       d,
             0,   0,        0,        1;

        T_k = T_k * T;
    }

    return Eigen::Vector3d(T_k(0, 3), T_k(1, 3), T_k(2, 3));
}

Eigen::Matrix<double, 6, Eigen::Dynamic> Jacobian(
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& dq,
    const Eigen::MatrixXd& DH,
    const Eigen::Vector3d& A_k)
{
    int n = static_cast<int>(q.size());
    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, n);
    J.setZero();

    Eigen::Vector3d z0(0.0, 0.0, 1.0);
    Eigen::Vector3d p0(0.0, 0.0, 0.0);

    Eigen::Matrix3d R_k = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d T_k = Eigen::Matrix4d::Identity();

    const double pi = M_PI;

    // First joint (index 0)
    {
        Eigen::Vector3d zj = z0;
        Eigen::Vector3d pj = p0;
        Eigen::Vector3d p = A_k;

        Eigen::Vector3d Jk = (zj.cross(p - pj));
        J.col(0).head(3) = Jk;
        J.col(0).tail(3) = zj;
    }

    for (int i = 0; i < n - 1; ++i) {
        double a     = DH(i, 0);
        double alpha = DH(i, 1);
        double d     = DH(i, 2);
        double theta_offset = DH(i, 3);

        double theta = q(i) + theta_offset;

        double ct = std::cos(theta);
        double st = std::sin(theta);
        double ca = std::cos(alpha);
        double sa = std::sin(alpha);

        Eigen::Matrix3d R;
        R << ct, -st * ca,  st * sa,
             st,  ct * ca, -ct * sa,
             0,   sa,       ca;

        Eigen::Matrix4d T;
        T << ct, -st * ca,  st * sa,  a * ct,
             st,  ct * ca, -ct * sa,  a * st,
             0,   sa,       ca,       d,
             0,   0,        0,        1;

        R_k = R_k * R;
        T_k = T_k * T;

        Eigen::Vector3d zj = R_k * z0;
        Eigen::Vector3d pj = T_k.block<3, 1>(0, 3);
        Eigen::Vector3d p = A_k;

        Eigen::Vector3d Jk = zj.cross(p - pj);
        J.col(i + 1).head(3) = Jk;
        J.col(i + 1).tail(3) = zj;
    }

    return J;
}

std::array<double, 4> compute_max_vel_capsule(
    const Eigen::Vector<double, 7>& q,
    const Eigen::Vector<double, 7>& q_p,
    const Eigen::Matrix<double, 7, 4>& DH,
    const std::array<double, 4>& rv,
    const Eigen::Vector3d& C_h,
    const Eigen::Vector3d& C_r)
{
    const double norm_C = (C_h - C_r).norm();
    if (norm_C < 1e-9) {
        // Degenerate case: points coincide; return zero velocities
        return {0.0, 0.0, 0.0, 0.0};
    }
    const Eigen::Vector3d C = (C_h - C_r) / norm_C;

    double vel_max_1 = 0.0;
    double vel_max_2 = 0.0;
    double vel_max_3 = 0.0;
    double vel_max_4 = 0.0;

    // Capsule 1-0
    {
        Eigen::Vector3d A_1 = compute_capsule_end_point(DH, q, 1);
        Eigen::Vector3d B_1 = Eigen::Vector3d::Zero();

        Eigen::Matrix<double, 6, 1> J1 = Jacobian(q.head(1), q_p.head(1), DH.topRows(1), A_1);
        Eigen::Matrix<double, 3, 1> Jp1 = J1.topRows(3);
        Eigen::Matrix<double, 3, 1> Jo1 = J1.bottomRows(3);

        Eigen::Vector3d A_1_p = Jp1 * q_p.head(1);
        Eigen::Vector3d omega_1 = Jo1 * q_p.head(1);

        double norm_A1B1 = (A_1 - B_1).norm();
        if (norm_A1B1 < 1e-9) norm_A1B1 = 1e-9;

        Eigen::Vector3d dir_AB = (A_1 - B_1) / norm_A1B1;
        Eigen::Vector3d dir_BA = (B_1 - A_1) / norm_A1B1;

        Eigen::Vector3d A_e1_p = A_1_p + omega_1.cross(dir_AB * rv[0]);
        Eigen::Vector3d B_e1_p = A_1_p + omega_1.cross(dir_BA * (norm_A1B1 + rv[0]));

        double vel_A = A_e1_p.dot(C);
        double vel_B = B_e1_p.dot(C);
        vel_max_1 = std::max(vel_A, vel_B);
        if (vel_max_1 < 0.0) vel_max_1 = 0.0;
    }

    // Capsule 2-1
    {
        Eigen::Vector3d A_2 = compute_capsule_end_point(DH, q, 3);
        Eigen::Vector3d B_2 = compute_capsule_end_point(DH, q, 2);

        Eigen::Matrix<double, 6, Eigen::Dynamic> J2 =
            Jacobian(q.head(3), q_p.head(3), DH.topRows(3), A_2);
        Eigen::Matrix<double, 3, Eigen::Dynamic> Jp2 = J2.topRows(3);
        Eigen::Matrix<double, 3, Eigen::Dynamic> Jo2 = J2.bottomRows(3);

        Eigen::Vector3d A_2_p = Jp2 * q_p.head(3);
        Eigen::Vector3d omega_2 = Jo2 * q_p.head(3);

        double norm_A2B2 = (A_2 - B_2).norm();
        if (norm_A2B2 < 1e-9) norm_A2B2 = 1e-9;

        Eigen::Vector3d dir_AB = (A_2 - B_2) / norm_A2B2;
        Eigen::Vector3d dir_BA = (B_2 - A_2) / norm_A2B2;

        Eigen::Vector3d A_e2_p = A_2_p + omega_2.cross(dir_AB * rv[1]);
        Eigen::Vector3d B_e2_p = A_2_p + omega_2.cross(dir_BA * (norm_A2B2 + rv[1]));

        double vel_A = A_e2_p.dot(C);
        double vel_B = B_e2_p.dot(C);
        vel_max_2 = std::max(vel_A, vel_B);
        if (vel_max_2 < 0.0) vel_max_2 = 0.0;
    }

    // Capsule 3-2
    {
        Eigen::Vector3d A_3 = compute_capsule_end_point(DH, q, 5);
        Eigen::Vector3d B_3 = compute_capsule_end_point(DH, q, 4);

        Eigen::Matrix<double, 6, Eigen::Dynamic> J3 =
            Jacobian(q.head(5), q_p.head(5), DH.topRows(5), A_3);
        Eigen::Matrix<double, 3, Eigen::Dynamic> Jp3 = J3.topRows(3);
        Eigen::Matrix<double, 3, Eigen::Dynamic> Jo3 = J3.bottomRows(3);

        Eigen::Vector3d A_3_p = Jp3 * q_p.head(5);
        Eigen::Vector3d omega_3 = Jo3 * q_p.head(5);

        double norm_A3B3 = (A_3 - B_3).norm();
        if (norm_A3B3 < 1e-9) norm_A3B3 = 1e-9;

        Eigen::Vector3d dir_AB = (A_3 - B_3) / norm_A3B3;
        Eigen::Vector3d dir_BA = (B_3 - A_3) / norm_A3B3;

        Eigen::Vector3d A_e3_p = A_3_p + omega_3.cross(dir_AB * rv[2]);
        Eigen::Vector3d B_e3_p = A_3_p + omega_3.cross(dir_BA * (norm_A3B3 + rv[2]));

        double vel_A = A_e3_p.dot(C);
        double vel_B = B_e3_p.dot(C);
        vel_max_3 = std::max(vel_A, vel_B);
        if (vel_max_3 < 0.0) vel_max_3 = 0.0;
    }

    // Capsule 5-4 (actually 7-6 in 1-based indexing, but 0-based here: 7 joints total)
    {
        Eigen::Vector3d A_4 = compute_capsule_end_point(DH, q, 7);
        Eigen::Vector3d B_4 = compute_capsule_end_point(DH, q, 6);

        Eigen::Matrix<double, 6, Eigen::Dynamic> J4 =
            Jacobian(q.head(7), q_p.head(7), DH.topRows(7), A_4);
        Eigen::Matrix<double, 3, Eigen::Dynamic> Jp4 = J4.topRows(3);
        Eigen::Matrix<double, 3, Eigen::Dynamic> Jo4 = J4.bottomRows(3);

        Eigen::Vector3d A_4_p = Jp4 * q_p.head(7);
        Eigen::Vector3d omega_4 = Jo4 * q_p.head(7);

        double norm_A4B4 = (A_4 - B_4).norm();
        if (norm_A4B4 < 1e-9) norm_A4B4 = 1e-9;

        Eigen::Vector3d dir_AB = (A_4 - B_4) / norm_A4B4;
        Eigen::Vector3d dir_BA = (B_4 - A_4) / norm_A4B4;

        Eigen::Vector3d A_e4_p = A_4_p + omega_4.cross(dir_AB * rv[3]);
        Eigen::Vector3d B_e4_p = A_4_p + omega_4.cross(dir_BA * (norm_A4B4 + rv[3]));

        double vel_A = A_e4_p.dot(C);
        double vel_B = B_e4_p.dot(C);
        vel_max_4 = std::max(vel_A, vel_B);
        if (vel_max_4 < 0.0) vel_max_4 = 0.0;
    }

    return {vel_max_1, vel_max_2, vel_max_3, vel_max_4};
}

std::array<std::array<Eigen::Vector3d, 2>, 4> compute_robot_capsules(
    const Eigen::Vector<double, 7>& q,
    const Eigen::Matrix<double, 7, 4>& DH,
    const Eigen::Vector3d& robot_to_marker_pos)
{
    std::array<std::array<Eigen::Vector3d, 2>, 4> capsules;

    // Capsule 1-0
    capsules[0][0] = compute_capsule_end_point(DH, q, 1) - robot_to_marker_pos;
    capsules[0][1] = Eigen::Vector3d::Zero() - robot_to_marker_pos;

    // Capsule 2-1
    capsules[1][0] = compute_capsule_end_point(DH, q, 3) - robot_to_marker_pos;
    capsules[1][1] = compute_capsule_end_point(DH, q, 2) - robot_to_marker_pos;

    // Capsule 3-2
    capsules[2][0] = compute_capsule_end_point(DH, q, 5) - robot_to_marker_pos;
    capsules[2][1] = compute_capsule_end_point(DH, q, 4) - robot_to_marker_pos;

    // Capsule 5-4 (7-6 in 0-based)
    capsules[3][0] = compute_capsule_end_point(DH, q, 7) - robot_to_marker_pos;
    capsules[3][1] = compute_capsule_end_point(DH, q, 6) - robot_to_marker_pos;

    return capsules;
}

SegmentDistanceResult distance_to_segment(
    const Eigen::Vector3d& P1,
    const Eigen::Vector3d& Q1,
    const Eigen::Vector3d& P2,
    const Eigen::Vector3d& Q2)
{
    Eigen::Vector3d D1 = Q1 - P1;
    Eigen::Vector3d D2 = Q2 - P2;
    Eigen::Vector3d R  = P1 - P2;

    double a = D1.dot(D1);
    double b = D1.dot(D2);
    double c = D1.dot(R);
    double e = D2.dot(D2);
    double f = D2.dot(R);

    double d = a * e - b * b;
    if (std::abs(d) < 1e-12) {
        // Segments are nearly parallel; fallback to endpoints
        double dist_min = std::numeric_limits<double>::infinity();
        Eigen::Vector3d C_h_min = P1;
        Eigen::Vector3d C_r_min = P2;

        const std::array<Eigen::Vector3d, 2> pts1 = {P1, Q1};
        const std::array<Eigen::Vector3d, 2> pts2 = {P2, Q2};

        for (const auto& p1 : pts1) {
            for (const auto& p2 : pts2) {
                double dist = (p1 - p2).norm();
                if (dist < dist_min) {
                    dist_min = dist;
                    C_h_min = p1;
                    C_r_min = p2;
                }
            }
        }

        SegmentDistanceResult res;
        res.distance = dist_min;
        res.C_h = C_h_min;
        res.C_r = C_r_min;
        return res;
    }

    double s = clamp((b * f - c * e) / d);
    double t = (b * s + f) / e;

    if (t < 0.0) {
        t = 0.0;
        s = clamp(-c / a);
    } else if (t > 1.0) {
        t = 1.0;
        s = clamp((b - c) / a);
    }

    Eigen::Vector3d C_h = P1 + D1 * s;
    Eigen::Vector3d C_r = P2 + D2 * t;
    double distance = (C_h - C_r).norm();

    SegmentDistanceResult res;
    res.distance = distance;
    res.C_h = C_h;
    res.C_r = C_r;
    return res;
}

std::optional<SkeletonDistanceResult> distance_to_skeleton(
    const std::array<Eigen::Vector3d, 15>& skeleton,
    const Eigen::Vector3d& P2,
    const Eigen::Vector3d& Q2)
{
    bool first = true;
    double min_dist = 0.0;
    Eigen::Vector3d C_h_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d C_r_min = Eigen::Vector3d::Zero();
    int ind_h = -1;

    for (size_t i = 0; i < skel_index.size(); ++i) {
        int idx1 = skel_index[i][0];
        int idx2 = skel_index[i][1];

        const Eigen::Vector3d& P1 = skeleton[idx1];
        const Eigen::Vector3d& Q1 = skeleton[idx2];

        // Skip segments with NaNs
        if (!P1.allFinite() || !Q1.allFinite()) {
            continue;
        }

        SegmentDistanceResult dist = distance_to_segment(P1, Q1, P2, Q2);

        if (first) {
            min_dist = dist.distance;
            C_h_min = dist.C_h;
            C_r_min = dist.C_r;
            ind_h = static_cast<int>(i);
            first = false;
        } else {
            if (dist.distance < min_dist) {
                min_dist = dist.distance;
                C_h_min = dist.C_h;
                C_r_min = dist.C_r;
                ind_h = static_cast<int>(i);
            }
        }
    }

    if (first) {
        // No valid segment found
        return std::nullopt;
    }

    SkeletonDistanceResult res;
    res.distance = min_dist;
    res.C_h = C_h_min;
    res.C_r = C_r_min;
    res.ind_h = ind_h;
    return res;
}

std::array<double, 4> capsule_calculation(
    double T_stop,
    const Eigen::Vector<double, 7>& q,
    const Eigen::Vector<double, 7>& q_p,
    const Eigen::Vector<double, 7>& q_pp,
    const Eigen::Vector3d& C_h,
    const Eigen::Vector3d& C_r,
    const Eigen::Matrix<double, 7, 4>& DH,
    const std::array<double, 4>& rv)
{
    const double Sh = max_human_speed * (T_reaction + T_stop);

    // Sample every 5 ms along the stop trajectory
    const double time_step = 0.005; // 5 ms
    const int n_samples = static_cast<int>(std::floor(T_stop / time_step)) + 1;

    // Precompute trajectories for each joint
    std::array<Eigen::VectorXd, 7> qi_arr;
    std::array<Eigen::VectorXd, 7> qi_p_arr;
    std::array<Eigen::VectorXd, 7> qi_pp_arr;

    for (int k = 0; k < 7; ++k) {
        auto traj = TrajPoly5(
            q(k), q_p(k), q_pp(k),
            q(k), 0.0, 0.0,
            T_stop, time_step);

        qi_arr[k] = traj.row(0);
        qi_p_arr[k] = traj.row(1);
        qi_pp_arr[k] = traj.row(2);
    }

    int N = static_cast<int>(qi_arr[0].size());

    std::vector<double> vel_max_1(N);
    std::vector<double> vel_max_2(N);
    std::vector<double> vel_max_3(N);
    std::vector<double> vel_max_4(N);

    // Time vector for integration
    Eigen::VectorXd dt_stop(N);
    for (int i = 0; i < N; ++i) {
        dt_stop(i) = (i == 0) ? 0.0 : (T_stop * static_cast<double>(i) / static_cast<double>(N - 1));
    }

    for (int m = 0; m < N; ++m) {
                Eigen::Vector<double, 7> q_stop;
        Eigen::Vector<double, 7> qp_stop;

        for (int k = 0; k < 7; ++k) {
            q_stop(k)  = qi_arr[k](m);
            qp_stop(k) = qi_p_arr[k](m);
        }

        auto vels = compute_max_vel_capsule(q_stop, qp_stop, DH, rv, C_h, C_r);
        vel_max_1[m] = vels[0];
        vel_max_2[m] = vels[1];
        vel_max_3[m] = vels[2];
        vel_max_4[m] = vels[3];
    }

    // Reaction-distance component (using initial max velocities)
    double Sr_1 = vel_max_1[0] * T_reaction;
    double Sr_2 = vel_max_2[0] * T_reaction;
    double Sr_3 = vel_max_3[0] * T_reaction;
    double Sr_4 = vel_max_4[0] * T_reaction;

    // Integrate along stop trajectory
    double Ss_1 = trapz(dt_stop, vel_max_1);
    double Ss_2 = trapz(dt_stop, vel_max_2);
    double Ss_3 = trapz(dt_stop, vel_max_3);
    double Ss_4 = trapz(dt_stop, vel_max_4);

    // Compute safety radii for each capsule
    double r_sw_1 = rv[0] + Sh + Sr_1 + Ss_1 + csi;
    double r_sw_2 = rv[1] + Sh + Sr_2 + Ss_2 + csi;
    double r_sw_3 = rv[2] + Sh + Sr_3 + Ss_3 + csi;
    double r_sw_4 = rv[3] + Sh + Sr_4 + Ss_4 + csi;

    return {r_sw_1, r_sw_2, r_sw_3, r_sw_4};
}

bool FlagStopServer(
    const std::array<Eigen::Vector3d, 15>& skeleton,
    const Eigen::Vector<double, 7>& q,
    const Eigen::Vector<double, 7>& q_p,
    const Eigen::Vector<double, 7>& q_pp,
    double T_stop,
    const Eigen::Matrix<double, 7, 4>& DH,
    const std::array<double, 4>& rv,
    const std::array<double, 10>& r_sw_h,
    const Eigen::Vector3d& robot_to_marker_pos)
{
    // Compute robot capsules
    auto robot_capsules = compute_robot_capsules(q, DH, robot_to_marker_pos);

    bool first = true;
    double min_distance = 0.0;
    Eigen::Vector3d C_h_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d C_r_min = Eigen::Vector3d::Zero();
    int ind_h = -1;
    int ind_r = -1;

    // Find closest robot capsule to any human skeleton segment
    for (int i = 0; i < 4; ++i) {
        const Eigen::Vector3d& P2 = robot_capsules[i][0];
        const Eigen::Vector3d& Q2 = robot_capsules[i][1];

        auto dist_opt = distance_to_skeleton(skeleton, P2, Q2);
        if (!dist_opt.has_value()) {
            continue;
        }

        const auto& dist = dist_opt.value();

        if (first) {
            min_distance = dist.distance;
            C_h_min = dist.C_h;
            C_r_min = dist.C_r;
            ind_h = dist.ind_h;
            ind_r = i;
            first = false;
        } else {
            if (dist.distance < min_distance) {
                min_distance = dist.distance;
                C_h_min = dist.C_h;
                C_r_min = dist.C_r;
                ind_h = dist.ind_h;
                ind_r = i;
            }
        }
    }

    if (first) {
        // No valid human segment found → no collision risk detected
        return false;
    }

    // Compute dynamic safety radii for robot capsules
    auto r_sw_r = capsule_calculation(
        T_stop, q, q_p, q_pp, C_h_min, C_r_min, DH, rv);

    // Collision condition:
    // if distance < human_radius + robot_radius → stop
    if (min_distance < (r_sw_h[ind_h] + r_sw_r[ind_r])) {
        return true;
    }

    return false;
}

} // namespace collision_avoidance