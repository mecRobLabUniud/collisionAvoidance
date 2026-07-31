#pragma once
#include <random>
#include "types.hpp"

namespace appendices {

struct Rng {
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform01{0.0, 1.0};

    Rng(unsigned seed = 42) : gen(seed) {}

    double rand01() { return uniform01(gen); }
    double randRange(double a, double b) { return a + rand01() * (b - a); }

    Vector3 randVector3(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) {
        return Vector3{randRange(xmin, xmax), randRange(ymin, ymax), randRange(zmin, zmax)};
    }
};

} // namespace appendices