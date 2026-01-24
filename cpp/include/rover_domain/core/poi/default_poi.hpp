#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_POI
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_POI

#include <rover_domain/core/poi/ipoi.hpp>
#include <rover_domain/utilities/math/cartesian.hpp>

namespace rover_domain {

/*
 *
 * Default boilerplate poi
 *
 */
template <typename ConstraintPolicy>
class DefaultPOI final : public IPOI {
   public:
    DefaultPOI(double value = 1.0, double obs_radius = 1.0, double capture_radius = -1.0,
        ConstraintPolicy constraint = ConstraintPolicy())
        : IPOI(value, obs_radius, capture_radius), m_constraint(constraint) {}

    [[nodiscard]] double constraint_satisfied(const POIs& pois, const Agents& agents, int poi_idx) const override {
        return m_constraint.is_satisfied(pois, agents, poi_idx);
    }

   private:
    ConstraintPolicy m_constraint;
};
}  // namespace rover_domain

#endif