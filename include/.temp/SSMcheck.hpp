#pragma once
#include "RobotModel.hpp"
#include "JacobianUtils.hpp"
#include "MinDistance.hpp"

// -----------------------------------------------------------------------
// Appendix F: "SSMcheck.m"
//
// function [exitflag] = SSMcheck(robot,delta_t,stopping_time,q_t,qdot_t,
//                                 qddot,ro,vo,delta)
//
// Checks whether a *given* (already chosen) qddot satisfies the ten SSM
// pairwise-segment constraints for the five consecutive links
// base->shoulder->forearm->wrist1->wrist2->tool0, i.e. whether it is safe
// to apply, without doing any optimization.
//
// NOTE: several lines in the original thesis code appear to contain
// copy-paste slips (e.g. row 5 of `b` references `(ro-r5)` where the
// pattern elsewhere would suggest `(ro-r2)`; row 10 references `J5`
// where the pattern elsewhere would suggest `J4`). These are preserved
// verbatim below and flagged inline with "[sic]" so the translation
// matches the source exactly; fix them if they turn out to be bugs.
// -----------------------------------------------------------------------
inline bool SSMcheck(const RobotModel& robot,
                      double delta_t,
                      double stopping_time,
                      const Eigen::VectorXd& q_t,
                      const Eigen::VectorXd& qdot_t,
                      const Eigen::VectorXd& qddot,
                      Eigen::Vector3d ro,
                      const Eigen::Vector3d& vo,
                      double delta) {
    Eigen::VectorXd q_tp = q_t + qdot_t * delta_t;

    Eigen::MatrixXd J_t  = linearBlock(robot.geometricJacobian(q_t, links::tool0));
    Eigen::MatrixXd Jd_t = linearBlock(derivativeJacobian(robot, q_t, links::tool0));
    (void)J_t; (void)Jd_t; // computed in the source but unused in SSMcheck itself

    ro = ro + delta_t * vo;

    // Per-link Jacobians (linear part), Jacobian time-derivatives, and
    // predicted positions at q_tp.
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

    const int n = robot.numJoints();
    Eigen::MatrixXd A(10, n);
    Eigen::VectorXd b(10);

    A.row(0) = (ro.transpose() * J5 - r5.transpose() * J5) * stopping_time * delta_t;
    A.row(1) = (ro.transpose() * J6 - r5.transpose() * J6 - (r6 - r5).transpose() * J5) * stopping_time * delta_t;
    A.row(2) = (ro.transpose() * J1 - r1.transpose() * J1) * stopping_time * delta_t;
    A.row(3) = (ro.transpose() * J2 - r1.transpose() * J2 - (r2 - r1).transpose() * J1) * stopping_time * delta_t;
    A.row(4) = (ro.transpose() * J2 - r2.transpose() * J2) * stopping_time * delta_t;
    A.row(5) = (ro.transpose() * J3 - r2.transpose() * J3 - (r3 - r2).transpose() * J2) * stopping_time * delta_t;
    A.row(6) = (ro.transpose() * J3 - r3.transpose() * J3) * stopping_time * delta_t;
    A.row(7) = (ro.transpose() * J4 - r3.transpose() * J4 - (r4 - r3).transpose() * J3) * stopping_time * delta_t;
    A.row(8) = (ro.transpose() * J4 - r4.transpose() * J4) * stopping_time * delta_t;
    A.row(9) = (ro.transpose() * J5 - r4.transpose() * J5 - (r5 - r4).transpose() * J4) * stopping_time * delta_t;

    b(0) = minsSSM(r5, r6, ro, delta) * minsSSM(r5, r6, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J5 - r5.transpose() * J5) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r5).transpose() * J5d * qdot_t * stopping_time).value();

    b(1) = minsSSM(r5, r6, ro, delta) * minsSSM(r5, r6, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J6 - r5.transpose() * J6 - (r6 - r5).transpose() * J6) * stopping_time * qdot_t).value() // [sic] pattern elsewhere uses J5 here
           - (delta_t * (ro - r6).transpose() * J6d * qdot_t * stopping_time).value();

    b(2) = minsSSM(r1, r2, ro, delta) * minsSSM(r1, r2, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J1 - r1.transpose() * J1) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r1).transpose() * J1d * qdot_t * stopping_time).value();

    b(3) = minsSSM(r1, r2, ro, delta) * minsSSM(r1, r2, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J2 - r1.transpose() * J2 - (r2 - r1).transpose() * J1) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r2).transpose() * J2d * qdot_t * stopping_time).value();

    b(4) = minsSSM(r2, r3, ro, delta) * minsSSM(r2, r3, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J2 - r2.transpose() * J2) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r5).transpose() * J2d * qdot_t * stopping_time).value(); // [sic] source uses (ro-r5), not (ro-r2)

    b(5) = minsSSM(r2, r3, ro, delta) * minsSSM(r2, r3, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J3 - r2.transpose() * J3 - (r3 - r2).transpose() * J2) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r3).transpose() * J3d * qdot_t * stopping_time).value();

    b(6) = minsSSM(r3, r4, ro, delta) * minsSSM(r3, r4, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J3 - r3.transpose() * J3) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r3).transpose() * J3d * qdot_t * stopping_time).value();

    b(7) = minsSSM(r3, r4, ro, delta) * minsSSM(r3, r4, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J4 - r3.transpose() * J4 - (r4 - r3).transpose() * J3) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r4).transpose() * J4d * qdot_t * stopping_time).value();

    b(8) = minsSSM(r4, r5, ro, delta) * minsSSM(r4, r5, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J4 - r4.transpose() * J4) * stopping_time * qdot_t).value()
           - (delta_t * (ro - r4).transpose() * J4d * qdot_t * stopping_time).value();

    b(9) = minsSSM(r4, r5, ro, delta) * minsSSM(r4, r5, ro, delta) - delta * delta / 4.0
           - ((ro.transpose() * J5 - r4.transpose() * J5 - (r5 - r4).transpose() * J5) * stopping_time * qdot_t).value() // [sic] pattern elsewhere uses J4 here
           - (delta_t * (ro - r5).transpose() * J5d * qdot_t * stopping_time).value();

    // check_matrix = A*qddot - b <= 0; exitflag = 1 iff all 10 hold.
    Eigen::VectorXd lhs = A * qddot - b;
    return (lhs.array() <= 0.0).all();
}
