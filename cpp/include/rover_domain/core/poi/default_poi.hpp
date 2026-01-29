#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI

#include <rover_domain/core/interface/ipoi.hpp>

namespace rover_domain {

/*
 *
 * Default boilerplate poi
 *
 */
template <typename Objective>
class DefaultPOI final : public IPOI {
   public:
    DefaultPOI(double value = 1.0, double obs_radius = 1.0, double capture_radius = -1.0,
        Objective objective = Objective())
        : IPOI(value, obs_radius, capture_radius), m_objective(objective) {}

    [[nodiscard]] double score(const POIs& pois, const Agents& agents, int poi_idx) const override {
        return m_objective.score(pois, agents, poi_idx);
    }

   private:
    Objective m_objective;
};
}  // namespace rover_domain

#endif
