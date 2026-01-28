#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_CLOSEST_COMPOSITION
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_CLOSEST_COMPOSITION

namespace rover_domain {

/*
 *
 * Sensor composition strategy
 * Take the highest value
 *
 */
class Closest {
   public:
    // template <std::ranges::range Range, typename Tp, typename Up>
    template <typename Range, typename Tp, typename Up>
    inline Tp compose(const Range& range, Tp, Up) const {
        return *std::max_element(std::begin(range), std::end(range));
    }
};

}  // namespace rover_domain

#endif
