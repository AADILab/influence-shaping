#ifndef BASIL_MATH_NORMS
#define BASIL_MATH_NORMS

#include <cmath>
#include <rover_domain/utilities/math/cartesian.hpp>

namespace thyme::math {

inline double l2_norm(const Point& a, const Point& b) {
    double x = b.x - a.x;
    double y = b.y - a.y;
    // Static cast to <long double> is to make intellisense happy
    return std::sqrt(static_cast<long double>(x * x + y * y));
}

inline std::pair<double, double> l2a(const Point& a, const Point& b) {
    // Match Python: vec = pos1 - pos0, angle = np.arctan2(vec[1], vec[0])
    auto dx = b.x - a.x;
    auto dy = b.y - a.y;
    auto angle = std::atan2(dy, dx) * to_degrees;
    auto distance = l2_norm(a, b);
    return {angle, distance};
}

}  // namespace thyme::math

#endif