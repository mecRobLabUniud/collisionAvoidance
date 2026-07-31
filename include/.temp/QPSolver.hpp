#pragma once
#include <Eigen/Dense>

// -----------------------------------------------------------------------
// Stands in for MATLAB's quadprog, as called in SSMPFL.m / SSMescape.m:
//
//   [qddot, fl, exitflag] = quadprog(H, f, A, b, [], [], q_lb, q_ub, ...
//                                     [0;0;0;0;0;0], options);
//
// i.e. solve:  minimize   0.5 x'Hx + f'x
//              subject to A x <= b
//                         lb <= x <= ub
//
// exitflag follows the MATLAB convention: 1 means a solution was found
// (first-order optimality conditions satisfied), anything else means the
// solver failed / was infeasible.
//
// This header only declares the interface. Wire it up to a real QP
// solver, e.g.:
//   - eiquadprog (header-only, Eigen-native, closest drop-in for quadprog)
//   - qpOASES
//   - OSQP (needs bounds/inequalities reformulated as a single sparse A)
//
// A minimal eiquadprog-based implementation would convert bounds into two
// extra block-diagonal inequality rows ( I*x <= ub, -I*x <= -lb ) and call
// solve_quadprog(H, f, CE, ce0, CI, ci0, x). Left out here so this header
// has no external dependency beyond Eigen.
// -----------------------------------------------------------------------
struct QPResult {
    Eigen::VectorXd x;
    int exitflag = 0;
};

QPResult solveQP(const Eigen::MatrixXd& H,
                  const Eigen::VectorXd& f,
                  const Eigen::MatrixXd& A,
                  const Eigen::VectorXd& b,
                  const Eigen::VectorXd& lb,
                  const Eigen::VectorXd& ub);
