#pragma once
#include "RobotModel.hpp"
#include "MinDistance.hpp"

// -----------------------------------------------------------------------
// Appendix G: "COLLcheck.m"
//
// function exitflag = COLLcheck(robot, q_tp, ro, HR_clearance)
//     r1 = getTransform(robot,q_tp,'base_link')(1:3,4);
//     r2 = getTransform(robot,q_tp,'shoulder_link')(1:3,4);
//     r3 = getTransform(robot,q_tp,'forearm_link')(1:3,4);
//     r4 = getTransform(robot,q_tp,'wrist_1_link')(1:3,4);
//     r5 = getTransform(robot,q_tp,'wrist_2_link')(1:3,4);
//     r6 = getTransform(robot,q_tp,'tool0')(1:3,4);
//
//     a(1) = minsros(r1,r2,ro) >= HR_clearance;
//     a(2) = minsros(r2,r3,ro) >= HR_clearance;
//     a(3) = minsros(r3,r4,ro) >= HR_clearance;
//     a(4) = minsros(r4,r5,ro) >= HR_clearance;
//     a(5) = minsros(r5,r6,ro) >= HR_clearance;
//     mins = min(a);
//
//     if mins == 0, exitflag = 0; else, exitflag = 1; end
// end
//
// True (1) if every consecutive link segment keeps at least HR_clearance
// distance from the obstacle point `ro` at configuration q_tp.
// -----------------------------------------------------------------------
inline bool COLLcheck(const RobotModel& robot,
                       const Eigen::VectorXd& q_tp,
                       const Eigen::Vector3d& ro,
                       double HR_clearance) {
    Eigen::Vector3d r1 = translationOf(robot.getTransform(q_tp, links::base));
    Eigen::Vector3d r2 = translationOf(robot.getTransform(q_tp, links::shoulder));
    Eigen::Vector3d r3 = translationOf(robot.getTransform(q_tp, links::forearm));
    Eigen::Vector3d r4 = translationOf(robot.getTransform(q_tp, links::wrist1));
    Eigen::Vector3d r5 = translationOf(robot.getTransform(q_tp, links::wrist2));
    Eigen::Vector3d r6 = translationOf(robot.getTransform(q_tp, links::tool0));

    bool a1 = minsros(r1, r2, ro).minros >= HR_clearance;
    bool a2 = minsros(r2, r3, ro).minros >= HR_clearance;
    bool a3 = minsros(r3, r4, ro).minros >= HR_clearance;
    bool a4 = minsros(r4, r5, ro).minros >= HR_clearance;
    bool a5 = minsros(r5, r6, ro).minros >= HR_clearance;

    bool mins = a1 && a2 && a3 && a4 && a5; // min(a) == 0 iff any is false

    return mins; // exitflag: true (1) = collision-free, false (0) = violation
}
