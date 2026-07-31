#include "min_distance_calculation.hpp"

SegmentDistanceResult segm_to_segm_distance(
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& q1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& q2)
{
    Eigen::Vector3d d1 = q1 - p1;
    Eigen::Vector3d d2 = q2 - p2;
    Eigen::Vector3d r  = p1 - p2;

    double a = d1.dot(d1);
    double b = d1.dot(d2);
    double c = d1.dot(r);
    double e = d2.dot(d2);
    double f = d2.dot(r);

    double d = a * e - b * b;
    if (std::abs(d) < 1e-12) {
        // Segments are nearly parallel; fallback to endpoints
        double dist_min = std::numeric_limits<double>::infinity();
        Eigen::Vector3d c_h_min = p1;
        Eigen::Vector3d c_r_min = p2;

        const std::array<Eigen::Vector3d, 2> pts1 = {p1, q1};
        const std::array<Eigen::Vector3d, 2> pts2 = {p2, q2};

        for (const auto& pa : pts1) {
            for (const auto& pb : pts2) {
                double dist = (pa - pb).norm();
                if (dist < dist_min) {
                    dist_min = dist;
                    c_h_min = pa;
                    c_r_min = pb;
                }
            }
        }

        SegmentDistanceResult res;
        res.length = dist_min;
        res.c_h = c_h_min;
        res.c_r = c_r_min;
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

    Eigen::Vector3d c_h = p1 + d1 * s;
    Eigen::Vector3d c_r = p2 + d2 * t;
    double distance = (c_h - c_r).norm();

    SegmentDistanceResult res;
    res.length = distance;
    res.c_h = c_h;
    res.c_r = c_r;
    return res;
}


std::optional<MinDistanceResult> min_distance_calculation(
    const std::array<Eigen::Vector3d, 15>& skeleton,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& q2)
{
    bool first = true;
    double min_dist = 0.0;
    Eigen::Vector3d c_h_min = Eigen::Vector3d::Zero();
    Eigen::Vector3d c_r_min = Eigen::Vector3d::Zero();
    int ind_h = -1;

    for (size_t i = 0; i < kSkelIndex.size(); ++i) {
        int idx1 = kSkelIndex[i][0];
        int idx2 = kSkelIndex[i][1];

        const Eigen::Vector3d& p1 = skeleton[idx1];
        const Eigen::Vector3d& q1 = skeleton[idx2];

        // Skip segments with NaNs
        if (!p1.allFinite() || !q1.allFinite()) {
            continue;
        }

        SegmentDistanceResult dist = segm_to_segm_distance(p1, q1, p2, q2);

        if (first) {
            min_dist = dist.distance;
            c_h_min = dist.c_h;
            c_r_min = dist.c_r;
            ind_h = static_cast<int>(i);
            first = false;
        } else {
            if (dist.distance < min_dist) {
                min_dist = dist.distance;
                c_h_min = dist.c_h;
                c_r_min = dist.c_r;
                ind_h = static_cast<int>(i);
            }
        }
    }

    if (first) {
        // No valid segment found
        return std::nullopt;
    }

    MinDistanceResult res;
    res.distance = min_dist;
    res.c_h = c_h_min;
    res.c_r = c_r_min;
    res.ind_h = ind_h;
    return res;
}