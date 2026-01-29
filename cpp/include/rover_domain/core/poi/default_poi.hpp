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
    DefaultPOI(
        double value,
        double obs_radius,
        double capture_radius,
        VisibilityScope scope = VisibilityScope::ALL,
        Objective objective = Objective()
    ) : IPOI(value, obs_radius, capture_radius, scope), m_objective(objective) {}

    [[nodiscard]] double score(const POIs& pois, const Agents& agents, int poi_idx) const override {
        return m_objective.score(pois, agents, poi_idx);
    }

   private:
    Objective m_objective;
};
}  // namespace rover_domain

#endif
