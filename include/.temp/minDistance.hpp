#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

// -----------------------------------------------------------------------
// Appendix I: "minsros.m"
//
// function [minros,roa,rob,rba,s] = minsros(ra,rb,ro)
//     rba = rb - ra;
//     roa = ro - ra;
//     rob = ro - rb;
//     s   = dot(roa,rba)/dot(rba,rba);
//     rs  = ra + s*rba;
//     ros = ro - rs;
//     if s >= 0 && s <= 1
//         minros = sqrt(dot(ros,ros));
//     else
//         minros = min(sqrt(dot(roa,roa)), sqrt(dot(rob,rob)));
//     end
// end
//
// Minimum distance between point `ro` and the segment [ra, rb].
// -----------------------------------------------------------------------
struct MinRosResult {
    double minros;
    Eigen::Vector3d roa;
    Eigen::Vector3d rob;
    Eigen::Vector3d rba;
    double s;
};

inline MinRosResult minsros(const Eigen::Vector3d& ra,
                             const Eigen::Vector3d& rb,
                             const Eigen::Vector3d& ro) {
    MinRosResult out;
    out.rba = rb - ra;
    out.roa = ro - ra;
    out.rob = ro - rb;
    out.s = out.roa.dot(out.rba) / out.rba.dot(out.rba);

    Eigen::Vector3d rs = ra + out.s * out.rba;
    Eigen::Vector3d ros = ro - rs;

    if (out.s >= 0.0 && out.s <= 1.0) {
        out.minros = ros.norm();
    } else {
        out.minros = std::min(out.roa.norm(), out.rob.norm());
    }
    return out;
}

// -----------------------------------------------------------------------
// Appendix H: "minsSSM.m"
//
// function [minSSM] = minsSSM(ra,rb,ro,delta)
//     [minros,roa,rob,rba] = minsros(ra,rb,ro);
//     if minros >= delta/2
//         minSSM = minros - delta/2;
//     else
//         omega_0_prime = dot(roa,roa);
//         omega_1 = -2*dot(roa,rba);
//         omega_2 = dot(rba,rba);
//         omega_0 = omega_0_prime - delta^2/4;
//         sq_delta = sqrt(omega_1^2 - 4*omega_0*omega_2);
//         s1 = (-omega_1 + sq_delta)/(2*omega_2);
//         s2 = (-omega_1 - sq_delta)/(2*omega_2);
//         if (s1<=1 && s1>=0) || (s2<=1 && s2>=0)
//             minSSM = 0;
//         else
//             minSSM = min(abs(sqrt(dot(roa,roa))-delta/2), ...
//                           abs(sqrt(dot(rob,rob))-delta/2));
//         end
//     end
// end
//
// Signed-style clearance measure used by the SSM constraints: distance
// from the safety cylinder of radius delta/2 around the segment [ra,rb]
// to the obstacle point ro.
// -----------------------------------------------------------------------
inline double minsSSM(const Eigen::Vector3d& ra,
                       const Eigen::Vector3d& rb,
                       const Eigen::Vector3d& ro,
                       double delta) {
    MinRosResult m = minsros(ra, rb, ro);

    if (m.minros >= delta / 2.0) {
        return m.minros - delta / 2.0;
    }

    double omega_0_prime = m.roa.dot(m.roa);
    double omega_1 = -2.0 * m.roa.dot(m.rba);
    double omega_2 = m.rba.dot(m.rba);
    double omega_0 = omega_0_prime - (delta * delta) / 4.0;

    double disc = omega_1 * omega_1 - 4.0 * omega_0 * omega_2;
    double sq_delta = std::sqrt(std::max(disc, 0.0)); // guard: matches MATLAB's
                                                        // sqrt() behaviour for
                                                        // disc>=0 cases only

    double s1 = (-omega_1 + sq_delta) / (2.0 * omega_2);
    double s2 = (-omega_1 - sq_delta) / (2.0 * omega_2);

    bool s1_in = (s1 <= 1.0 && s1 >= 0.0);
    bool s2_in = (s2 <= 1.0 && s2 >= 0.0);

    if (s1_in || s2_in) {
        return 0.0;
    }

    return std::min(std::abs(m.roa.norm() - delta / 2.0),
                     std::abs(m.rob.norm() - delta / 2.0));
}
