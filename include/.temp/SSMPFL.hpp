#pragma once
#include "RobotModel.hpp"
#include "JacobianUtils.hpp"
#include "MinDistance.hpp"
#include "QPSolver.hpp"

// -----------------------------------------------------------------------
// Appendix D: "SSMPFL.m"
//
// function [qddot,qd_tplusone,q_tplusone,x_tplusone,xd_tplusone,exitflag] =
//     SSMPFL(robot,q_limits,qdot_limits,qddot_limits,delta_t,stopping_time,
//            q_t,qdot_t,x_ref_tplusone,xd_ref_tplusone,qddot_suggestion,
//            ro,vo,delta,q_des,qd_des,Qv)
//
// One-step QP: track a task-space + joint-space reference while enforcing
// joint/velocity/acceleration limits and the combined SSM+PFL safety
// constraints (Equation 2.70 in the thesis).
// -----------------------------------------------------------------------
struct SSMPFLResult {
    Eigen::VectorXd qddot;
    Eigen::VectorXd qd_tplusone;
    Eigen::VectorXd q_tplusone;
    Eigen::Vector3d x_tplusone;
    Eigen::Vector3d xd_tplusone;
    int exitflag;
};

inline SSMPFLResult SSMPFL(const RobotModel& robot,
                            const Eigen::MatrixXd& q_limits,    // n x 2 [min max]
                            const Eigen::MatrixXd& qdot_limits, // n x 2
                            const Eigen::MatrixXd& qddot_limits,// n x 2
                            double delta_t,
                            double stopping_time,
                            const Eigen::VectorXd& q_t,
                            const Eigen::VectorXd& qdot_t,
                            const Eigen::Vector3d& x_ref_tplusone,
                            const Eigen::Vector3d& xd_ref_tplusone,
                            const Eigen::VectorXd& /*qddot_suggestion*/, // unused in source too
                            Eigen::Vector3d ro,
                            const Eigen::Vector3d& vo,
                            double delta,
                            const Eigen::VectorXd& q_des,
                            const Eigen::VectorXd& /*qd_des*/, // unused in source too
                            double Qv) {
    const int n = robot.numJoints();

    Eigen::VectorXd q_tp = q_t + delta_t * qdot_t;

    Eigen::Vector3d x_t = translationOf(robot.getTransform(q_t, links::tool0));
    Eigen::MatrixXd J_t = linearBlock(robot.geometricJacobian(q_t, links::tool0));
    Eigen::MatrixXd Jd_t = linearBlock(derivativeJacobian(robot, q_t, links::tool0));
    Eigen::Vector3d xd_t = J_t * qdot_t;
    (void)xd_t; // computed in source, unused there too

    ro = ro + vo * delta_t;

    // --- Objective: joint-space + task-space tracking ------------------
    Eigen::MatrixXd weight_matrix = Eigen::MatrixXd::Zero(6, 6);
    weight_matrix(0, 0) = 1.5;
    weight_matrix(1, 1) = 3.0;
    weight_matrix(2, 2) = 3.0;
    weight_matrix(3, 3) = 1.75;
    weight_matrix(4, 4) = 1.75;
    weight_matrix(5, 5) = 0.1;
    // NOTE: weight_matrix is 6x6 in the source regardless of n; if your
    // robot doesn't have 6 joints, resize/adjust this block accordingly.

    double dt2 = delta_t * delta_t;
    double dt4 = dt2 * dt2;

    Eigen::MatrixXd Hp = 70.0 * dt4 / 4.0 * weight_matrix + 1.0 * dt4 / 2.0 * J_t.transpose() * J_t;

    Eigen::VectorXd kpp = (-x_ref_tplusone + dt2 / 2.0 * Jd_t * qdot_t + delta_t * J_t * qdot_t + x_t);
    Eigen::VectorXd fpp = dt2 * J_t.transpose() * kpp;

    Eigen::VectorXd kp = (-q_des + delta_t * qdot_t + q_t);
    Eigen::VectorXd fp = 70.0 * (dt2 / 2.0 * kp) + 1.0 * fpp;

    Eigen::MatrixXd Hv = dt2 * 2.0 * J_t.transpose() * J_t;
    Eigen::VectorXd kv = (xd_ref_tplusone - J_t * qdot_t - delta_t * Jd_t * qdot_t);
    Eigen::VectorXd fv = -delta_t * 2.0 * J_t.transpose() * kv;

    Eigen::MatrixXd H = Hp + Qv * Hv;
    Eigen::VectorXd f = fp + Qv * fv;

    // --- Kinematic / dynamic bounds ------------------------------------
    Eigen::VectorXd qmin = (q_limits.col(0) - q_t - delta_t * qdot_t) * 2.0 / dt2;
    Eigen::VectorXd qmax = (q_limits.col(1) - q_t - delta_t * qdot_t) * 2.0 / dt2;
    Eigen::VectorXd qdmin = (qdot_limits.col(0) - qdot_t) / delta_t;
    Eigen::VectorXd qdmax = (qdot_limits.col(1) - qdot_t) / delta_t;
    Eigen::VectorXd qddmin = qddot_limits.col(0);
    Eigen::VectorXd qddmax = qddot_limits.col(1);

    Eigen::VectorXd q_lb = qmin.cwiseMax(qdmin).cwiseMax(qddmin);
    Eigen::VectorXd q_ub = qmax.cwiseMin(qdmax).cwiseMin(qddmax);

    // --- SSM+PFL constraints (per-link kinematics) ----------------------
    Eigen::MatrixXd J1  = linearBlock(robot.geometricJacobian(q_t, links::base));
    Eigen::MatrixXd J1d = linearBlock(derivativeJacobian(robot, q_t, links::base));
    Eigen::Vector3d r1  = translationOf(robot.getTransform(q_tp, links::base));

    Eigen::MatrixXd J2  = linearBlock(robot.geometricJacobian(q_t, links::shoulder));
    Eigen::MatrixXd J2d = linearBlock(derivativeJacobian(robot, q_t, links::shoulder));
    Eigen::Vector3d r2  = translationOf(robot.getTransform(q_tp, links::shoulder));

    Eigen::MatrixXd J3  = linearBlock(robot.geometricJacobian(q_t, links::forearm));
    Eigen::MatrixXd J3d = linearBlock(derivativeJacobian(robot, q_t, links::forearm));
    Eigen::Vector3d r3  = translationOf(robot.getTransform(q_tp, links::forearm));

    Eigen::MatrixXd J4  = linearBlock(robot.geometricJacobian(q_t, links::wrist1));
    Eigen::MatrixXd J4d = linearBlock(derivativeJacobian(robot, q_t, links::wrist1));
    Eigen::Vector3d r4  = translationOf(robot.getTransform(q_tp, links::wrist1));

    Eigen::MatrixXd J5  = linearBlock(robot.geometricJacobian(q_t, links::wrist2));
    Eigen::MatrixXd J5d = linearBlock(derivativeJacobian(robot, q_t, links::wrist2));
    Eigen::Vector3d r5  = translationOf(robot.getTransform(q_tp, links::wrist2));

    Eigen::MatrixXd J6  = linearBlock(robot.geometricJacobian(q_t, links::tool0));
    Eigen::MatrixXd J6d = linearBlock(derivativeJacobian(robot, q_t, links::tool0));
    Eigen::Vector3d r6  = translationOf(robot.getTransform(q_tp, links::tool0));

    // NOTE: unlike SSMescape/SSMcheck, here the constraint rows are scaled
    // by delta_t only (not stopping_time), and the RHS margin term uses
    // 1/stopping_time instead — this matches Equation 2.70 in the source
    // (as opposed to Equation 2.52 used by SSMescape/SSMcheck).
    Eigen::MatrixXd A(10, n);
    Eigen::VectorXd b(10);

    A.row(0) = (ro.transpose() * J5 - r5.transpose() * J5) * delta_t;
    A.row(1) = (ro.transpose() * J6 - r5.transpose() * J6 - (r6 - r5).transpose() * J5) * delta_t;
    A.row(2) = (ro.transpose() * J1 - r1.transpose() * J1) * delta_t;
    A.row(3) = (ro.transpose() * J2 - r1.transpose() * J2 - (r2 - r1).transpose() * J1) * delta_t;
    A.row(4) = (ro.transpose() * J2 - r2.transpose() * J2) * delta_t;
    A.row(5) = (ro.transpose() * J3 - r2.transpose() * J3 - (r3 - r2).transpose() * J2) * delta_t;
    A.row(6) = (ro.transpose() * J3 - r3.transpose() * J3) * delta_t;
    A.row(7) = (ro.transpose() * J4 - r3.transpose() * J4 - (r4 - r3).transpose() * J3) * delta_t;
    A.row(8) = (ro.transpose() * J4 - r4.transpose() * J4) * delta_t;
    A.row(9) = (ro.transpose() * J5 - r4.transpose() * J5 - (r5 - r4).transpose() * J4) * delta_t;

    double d2 = delta * delta;

    b(0) = 1.0 / stopping_time * (std::pow(minsSSM(r5, r6, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J5 - r5.transpose() * J5) * qdot_t).value()
           - (delta_t * (ro - r5).transpose() * J5d * qdot_t).value();

    b(1) = 1.0 / stopping_time * (std::pow(minsSSM(r5, r6, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J6 - r5.transpose() * J6 - (r6 - r5).transpose() * J5) * qdot_t).value()
           - (delta_t * (ro - r6).transpose() * J6d * qdot_t).value();

    b(2) = 1.0 / stopping_time * (std::pow(minsSSM(r1, r2, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J1 - r1.transpose() * J1) * qdot_t).value()
           - (delta_t * (ro - r1).transpose() * J1d * qdot_t).value();

    b(3) = 1.0 / stopping_time * (std::pow(minsSSM(r1, r2, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J2 - r1.transpose() * J2 - (r2 - r1).transpose() * J1) * qdot_t).value()
           - (delta_t * (ro - r2).transpose() * J2d * qdot_t).value();

    b(4) = 1.0 / stopping_time * (std::pow(minsSSM(r2, r3, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J2 - r2.transpose() * J2) * qdot_t).value()
           - (delta_t * (ro - r2).transpose() * J2d * qdot_t).value();

    b(5) = 1.0 / stopping_time * (std::pow(minsSSM(r2, r3, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J3 - r2.transpose() * J3 - (r3 - r2).transpose() * J2) * qdot_t).value()
           - (delta_t * (ro - r3).transpose() * J3d * qdot_t).value();

    b(6) = 1.0 / stopping_time * (std::pow(minsSSM(r3, r4, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J3 - r3.transpose() * J3) * qdot_t).value()
           - (delta_t * (ro - r3).transpose() * J3d * qdot_t).value();

    b(7) = 1.0 / stopping_time * (std::pow(minsSSM(r3, r4, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J4 - r3.transpose() * J4 - (r4 - r3).transpose() * J3) * qdot_t).value()
           - (delta_t * (ro - r4).transpose() * J4d * qdot_t).value();

    b(8) = 1.0 / stopping_time * (std::pow(minsSSM(r4, r5, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J4 - r4.transpose() * J4) * qdot_t).value()
           - (delta_t * (ro - r4).transpose() * J4d * qdot_t).value();

    b(9) = 1.0 / stopping_time * (std::pow(minsSSM(r4, r5, ro, delta), 2) - d2 / 4.0)
           - ((ro.transpose() * J5 - r4.transpose() * J5 - (r5 - r4).transpose() * J4) * qdot_t).value()
           - (delta_t * (ro - r5).transpose() * J5d * qdot_t).value();

    // --- Optimization ----------------------------------------------------
    QPResult qp = solveQP(H, f, A, b, q_lb, q_ub);

    SSMPFLResult out;
    out.exitflag = qp.exitflag;

    if (qp.exitflag == 1) {
        out.qddot = qp.x;
        out.q_tplusone = q_t + delta_t * qdot_t + dt2 / 2.0 * out.qddot;
        out.qd_tplusone = qdot_t + delta_t * out.qddot;

        Eigen::Matrix4d x_tplusone_T = robot.getTransform(out.q_tplusone, links::tool0);
        out.x_tplusone = translationOf(x_tplusone_T);

        Eigen::MatrixXd J_tplusone = linearBlock(robot.geometricJacobian(out.q_tplusone, links::tool0));
        out.xd_tplusone = J_tplusone * out.qd_tplusone;
    } else {
        out.qddot = Eigen::VectorXd::Zero(n);
        out.q_tplusone = q_t;
        out.qd_tplusone = Eigen::VectorXd::Zero(n);
        out.x_tplusone = x_t;
        out.xd_tplusone = Eigen::Vector3d::Zero();
    }

    return out;
}
