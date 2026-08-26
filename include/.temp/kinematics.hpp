#pragma once
#include <eigen3/Eigen/Dense>
#include <array>
#include <cmath>
#include <iostream>
#include <chrono>

bool ee = false;

const double d1 = 0.3330;
const double d3 = 0.3160;
const double d5 = 0.3840;
const double d7 = 0.107;
const double d7e = 0.2104;
const double a4 = 0.0825;
const double a7 = 0.0880;
const std::array<double, 7> q_min = {{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
const std::array<double, 7> q_max = {{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};


const Eigen::Matrix<double, 7, 4> DH = []() {
    const double pi = M_PI;
    Eigen::Matrix<double, 7, 4> m;
    m << 0.0,      0.0,      0.333,  0.0,
         0.0,     -pi/2.0,   0.0,    0.0,
         0.0,      pi/2.0,   0.316,  0.0,
         0.0825,   pi/2.0,   0.0,    0.0,
        -0.0825,  -pi/2.0,   0.384,  0.0,
         0.0,      pi/2.0,   0.0,    0.0,
         0.088,    0.0,      0.107,  0.0;
    return m;
}();


void error() {
    std::cout << "Error: geometrical inconsistency." << std::endl;
}



void error(int n) {
    std::cout << "Error: Nan elements in the arrays in position " << n << std::endl;
}



void error(int n, double val) {
    std::cout << "Error: q" << n + 1 << " outside constraints." << std::endl;
    std::cout << "Actual joint value " << val << " must be included in range [" << q_min[n] << ", " << q_max[n] << "]." << std::endl;
}



std::array<std::array<double, 7>, 4> IK_solver(Eigen::Map< Eigen::Matrix<double, 4, 4> > O_T_EE, double q7, std::array<double, 7> q_actual_array, bool print) {
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::system_clock::now();

    const double d1 = 0.3330;
    const double d3 = 0.3160;
    const double d5 = 0.3840;
    const double d7 = 0.107;
    const double d7e = 0.2104;
    const double a4 = 0.0825;
    const double a7 = 0.0880;

    const double LL24 = 0.10666225;
    const double LL46 = 0.15426225;
    const double L24 = 0.326591870689;
    const double L46 = 0.392762332715;

    const double thetaH46 = 1.35916951803; 
    const double theta342 = 1.31542071191; 
    const double theta46H = 0.211626808766;
    
    const std::array< std::array<double, 7>, 4 > q_all_NAN = {{ {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
                                                                {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
                                                                {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}},
                                                                {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}} }};
    const std::array<double, 7> q_NAN = {{NAN, NAN, NAN, NAN, NAN, NAN, NAN}};
    std::array< std::array<double, 7>, 4 > q_all = q_all_NAN;
        
    if (q7 <= q_min[6] || q7 >= q_max[6]) {
        if (print) {
            error(6, q7);
        }
        return q_all_NAN;
    }
    else {
        for (int i = 0; i < 4; i++) {
            q_all[i][6] = q7;
        }
    }
    
    // compute p_6
    Eigen::Matrix3d R_EE = O_T_EE.topLeftCorner<3, 3>();
    Eigen::Vector3d z_EE = O_T_EE.block<3, 1>(0, 2);
    Eigen::Vector3d p_EE = O_T_EE.block<3, 1>(0, 3);
    double dee;
    if (ee) {
        dee = d7e;
    }
    else {
        dee = d7;
    } 
    Eigen::Vector3d p_7 = p_EE - d7e * z_EE;
    
    Eigen::Vector3d x_EE_6;
    x_EE_6 << std::cos(q7 - M_PI_4), -std::sin(q7 - M_PI_4), 0.0;
    Eigen::Vector3d x_6 = R_EE*x_EE_6;
    x_6 /= x_6.norm(); // visibly increases accuracy
    Eigen::Vector3d p_6 = p_7 - a7*x_6;
    
    // compute q4
    Eigen::Vector3d p_2;
    p_2 << 0.0, 0.0, d1;
    Eigen::Vector3d V26 = p_6 - p_2;
    
    double LL26 = V26[0]*V26[0] + V26[1]*V26[1] + V26[2]*V26[2];
    double L26 = std::sqrt(LL26);
    
    if (L24 + L46 < L26 || L24 + L26 < L46 || L26 + L46 < L24) {
        if (print) {
            error();
        }
        return q_all_NAN;
    }  
    
    double theta246 = std::acos((LL24 + LL46 - LL26)/2.0/L24/L46);
    double q4 = theta246 + thetaH46 + theta342 - 2.0*M_PI;
    if (q4 <= q_min[3] || q4 >= q_max[3]) {
        if (print) {
            error(3, q4);
        }
        return q_all_NAN;
    }
    else {
        for (int i = 0; i < 4; i++) {
            q_all[i][3] = q4;
        }
    }
    
    // compute q6
    double theta462 = std::acos((LL26 + LL46 - LL24)/2.0/L26/L46);
    double theta26H = theta46H + theta462;
    double D26 = -L26*std::cos(theta26H);
    
    Eigen::Vector3d Z_6 = z_EE.cross(x_6);
    Eigen::Vector3d Y_6 = Z_6.cross(x_6);
    Eigen::Matrix3d R_6;
    R_6.col(0) = x_6;
    R_6.col(1) = Y_6/Y_6.norm();
    R_6.col(2) = Z_6/Z_6.norm();
    Eigen::Vector3d V_6_62 = R_6.transpose()*(-V26);

    double Phi6 = std::atan2(V_6_62[1], V_6_62[0]);
    double Theta6 = std::asin(D26/std::sqrt(V_6_62[0]*V_6_62[0] + V_6_62[1]*V_6_62[1]));

    std::array<double, 2> q6;
    q6[0] = M_PI - Theta6 - Phi6;
    q6[1] = Theta6 - Phi6;
    
    for (int i = 0; i < 2; i++) {
        if (q6[i] <= q_min[5]) {
            q6[i] += 2.0*M_PI;
        }
        else if (q6[i] >= q_max[5]) {
            q6[i] -= 2.0*M_PI;
        }
        
        if (q6[i] <= q_min[5] || q6[i] >= q_max[5]) {
            q_all[2*i] = q_NAN;
            q_all[2*i + 1] = q_NAN;
        }
        else {
            q_all[2*i][5] = q6[i];
            q_all[2*i + 1][5] = q6[i];
        }
    }
    if (std::isnan(q_all[0][5]) && std::isnan(q_all[2][5])) {
        if (print) {
            error(5);
        }
        return q_all_NAN;
    }

    // compute q1 & q2
    double thetaP26 = 3.0*M_PI_2 - theta462 - theta246 - theta342;
    double thetaP = M_PI - thetaP26 - theta26H;
    double LP6 = L26*sin(thetaP26)/std::sin(thetaP);
    
    std::array< Eigen::Vector3d, 4 > z_5_all;
    std::array< Eigen::Vector3d, 4 > V2P_all;
    
    for (int i = 0; i < 2; i++) {
        Eigen::Vector3d z_6_5;
        z_6_5 << std::sin(q6[i]), std::cos(q6[i]), 0.0;
        Eigen::Vector3d z_5 = R_6*z_6_5;
        Eigen::Vector3d V2P = p_6 - LP6*z_5 - p_2;
        
        z_5_all[2*i] = z_5;
        z_5_all[2*i + 1] = z_5;
        V2P_all[2*i] = V2P;
        V2P_all[2*i + 1] = V2P;
        
        double L2P = V2P.norm();
        
        if (std::fabs(V2P[2]/L2P) > 0.999) {
            q_all[2*i][0] = q_actual_array[0];
            q_all[2*i][1] = 0.0;
            q_all[2*i + 1][0] = q_actual_array[0];
            q_all[2*i + 1][1] = 0.0;
        }
        else {
            q_all[2*i][0] = std::atan2(V2P[1], V2P[0]);
            q_all[2*i][1] = std::acos(V2P[2]/L2P);
            if (q_all[2*i][0] < 0) {
                q_all[2*i + 1][0] = q_all[2*i][0] + M_PI;
            }
            else {
                q_all[2*i + 1][0] = q_all[2*i][0] - M_PI;
            }
            q_all[2*i + 1][1] = -q_all[2*i][1];
        }
    }
    
    for (int i = 0; i < 4; i++) {
        if ( q_all[i][0] <= q_min[0] || q_all[i][0] >= q_max[0] || q_all[i][1] <= q_min[1] || q_all[i][1] >= q_max[1] ) {
            q_all[i] = q_NAN;
            if (print) {
                error(0, q_all[i][0]);
            }
            continue;
        }

        // compute q3
        Eigen::Vector3d z_3 = V2P_all[i]/V2P_all[i].norm();
        Eigen::Vector3d Y_3 = -V26.cross(V2P_all[i]);
        Eigen::Vector3d y_3 = Y_3/Y_3.norm();
        Eigen::Vector3d x_3 = y_3.cross(z_3);
        Eigen::Matrix3d R_1;
        double c1 = std::cos(q_all[i][0]);
        double s1 = std::sin(q_all[i][0]);
        R_1 <<   c1,  -s1,  0.0,
                 s1,   c1,  0.0,
                0.0,  0.0,  1.0;
        Eigen::Matrix3d R_1_2;
        double c2 = std::cos(q_all[i][1]);
        double s2 = std::sin(q_all[i][1]);
        R_1_2 <<   c2,  -s2, 0.0,
                  0.0,  0.0, 1.0,
                  -s2,  -c2, 0.0;
        Eigen::Matrix3d R_2 = R_1*R_1_2;
        Eigen::Vector3d x_2_3 = R_2.transpose()*x_3;
        q_all[i][2] = std::atan2(x_2_3[2], x_2_3[0]);
        
        if (q_all[i][2] <= q_min[2] || q_all[i][2] >= q_max[2]) {
            q_all[i] = q_NAN;
            if (print) {
                error(2, q_all[i][2]);
            }
            continue;
        }
        
        // compute q5
        Eigen::Vector3d VH4 = p_2 + d3*z_3 + a4*x_3 - p_6 + d5*z_5_all[i];
        Eigen::Matrix3d R_5_6;
        double c6 = std::cos(q_all[i][5]);
        double s6 = std::sin(q_all[i][5]);
        R_5_6 <<   c6,  -s6,  0.0,
                  0.0,  0.0, -1.0,
                   s6,   c6,  0.0;
        Eigen::Matrix3d R_5 = R_6*R_5_6.transpose();
        Eigen::Vector3d V_5_H4 = R_5.transpose()*VH4;
        
        q_all[i][4] = -std::atan2(V_5_H4[1], V_5_H4[0]);
        if (q_all[i][4] <= q_min[4] || q_all[i][4] >= q_max[4]) {
            q_all[i] = q_NAN;
            if (print) {
                error(4, q_all[i][4]);
            }
            continue;
        }
    }

    if (print) {
        std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::system_clock::now();
        std::chrono::duration<double> t_elaps = t_end - t_start;
        std::cout << std::endl << "Elapsed time for IK solver: " << t_elaps.count() << "s" << std::endl;
    }
    
    return q_all;
}



Eigen::Matrix4d FK_solver(std::array<double, 7> q, bool print) {
    std::chrono::time_point<std::chrono::system_clock> t_start = std::chrono::system_clock::now();

    q[6] -= M_PI_2;

    const double s0 = std::sin(q[0]), c0 = std::cos(q[0]);
    const double s1 = std::sin(q[1]), c1 = std::cos(q[1]);
    const double s2 = std::sin(q[2]), c2 = std::cos(q[2]);
    const double s3 = std::sin(q[3]), c3 = std::cos(q[3]);
    const double s4 = std::sin(q[4]), c4 = std::cos(q[4]);
    const double s5 = std::sin(q[5]), c5 = std::cos(q[5]);
    const double s6 = std::sin(q[6]), c6 = std::cos(q[6]);

    double c_p_s6 = c6 + s6;
    double c_m_s6 = c6 - s6;

    const double t1 = c3*(c5*c4*c_m_s6 + s4*c_p_s6) - s3*s5*c_m_s6;
    const double t3 = c3*(c5*c4*c_p_s6 - s4*c_m_s6) - s3*s5*c_p_s6;
    const double t2  = c4*c_p_s6 - c5*s4*c_m_s6;
    const double t18 = c4*c_m_s6 + c5*s4*c_p_s6;
    const double t20 = c3*s5*c_p_s6 - s3*s4*c_m_s6;
    const double t21 = c3*s5*c_m_s6 + s3*s4*c_p_s6;
    const double t4 = s1*(t20 + c5*c4*s3*c_p_s6) + c1*(c2*t3 - s2*t18);
    const double t5 = s1*(t21 + c5*c4*s3*c_m_s6) + c1*(c2*t1 + s2*t2);
    double dee;
    if (ee) {
        dee = d7e;
    }
    else {
        dee = d7;
    }
    const double t8 = -a7*c5 - d7e*s5;
    const double t22 = d5 + a7*s5 - d7e*c5;
    const double t6 = -a4 - c4*t8;
    const double t7 = d3 + c3*t22 + s3*t6;
    const double t9 = -a4 + s3*t22 - c3*t6;
    const double t13 = c1*t21 + c5*s4*s1*s2*c_m_s6;
    const double t14 = c1*t20 + c5*s4*s1*s2*c_p_s6;
    const double t15 = c2*t18 + s2*t3;
    const double t16 = c2*t2 - s2*t1;
    const double t17 = s1*s3 + c1*c2*c3;
    const double t24 = s1*c3 - c1*c2*s3;
    const double t19 = c1*(-c2*t9 + s2*s4*t8) + s1*t7;
    const double t23 = s2*t9 + c2*s4*t8;

    const double sq2 = 1./std::sqrt(2);

    std::array<double, 16> pose_EE = {{
        (s0*t16 + c0*t5)*sq2,
        (-c0*t16 + s0*t5)*sq2,
        (t13 - c4*(s1*s2*c_p_s6 - c1*c5*s3*c_m_s6) - c2*s1*t1)*sq2,
        0,
        -(-s0*t15 + c0*t4)*sq2,
        -(c0*t15 + s0*t4)*sq2,
        -(t14 + c4*(s1*s2*c_m_s6 + c1*c5*s3*c_p_s6) - c2*s1*t3)*sq2,
        0,
        -(c5*(c0*t24 + s3*s0*s2) + c4*s5*(c3*s0*s2 - c0*t17) + (c2*s0 + c0*c1*s2)*s4*s5),
        -(c5*(s0*t24 - s3*c0*s2) - c4*s5*(c3*c0*s2 + s0*t17) - (c2*c0 - s0*c1*s2)*s4*s5),
        -(c1*(c3*c5 - s3*c4*s5) + s1*(c2*(c5*s3 + c3*c4*s5) - s2*s4*s5)),
        0,
        s0*t23 + c0*t19,
        s0*t19 - c0*t23,
        d1 - s1*s2*s4*t8 + c2*s1*t9 + c1*t7,
        1,
    }};

    Eigen::Matrix4d O_T_EE(Eigen::Matrix4d(pose_EE.data()));

    if (print) {
        std::chrono::time_point<std::chrono::system_clock> t_end = std::chrono::system_clock::now();
        std::chrono::duration<double> t_elaps = t_end - t_start;
        std::cout << std::endl << "Elapsed time for FK solver: " << t_elaps.count() << "s" << std::endl;
    }

    return O_T_EE;
}


Eigen::Vector3d FK_end_point(
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