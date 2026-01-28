#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_DENSITY_COMPOSITION
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_DENSITY_COMPOSITION

namespace rover_domain {

/*
 *
 * Sensor composition strategy
 * Average across sensed values
 *
 */
class Density {
   public:
    // template <std::ranges::range Range, typename Tp, typename Up>
    template <typename Range, typename Tp, typename Up>
    inline Tp compose(const Range& range, Tp init, Up scale) const {
        return std::accumulate(std::begin(range), std::end(range), init) / scale;
    }
};

}  // namespace rover_domain

#endif
