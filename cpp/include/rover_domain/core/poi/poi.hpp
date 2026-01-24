#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_POI
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_POI

#include <rover_domain/core/detail/pack.hpp>
#include <rover_domain/utilities/math/cartesian.hpp>

namespace rover_domain {

/*
 *
 * Default boilerplate poi
 *
 */
template <typename ConstraintPolicy>
class POI final : public IPOI {
   public:
    POI(double value = 1.0, double obs_radius = 1.0, double capture_radius = -1.0,
        ConstraintPolicy constraint = ConstraintPolicy())
        : IPOI(value, obs_radius, capture_radius), m_constraint(constraint) {}

    [[nodiscard]] double constraint_satisfied(const EntityPack& entity_pack) const override {
        // std::cout << "POI::constraint_satisfied()" << std::endl;
        return m_constraint.is_satisfied(entity_pack);
    }

   private:
    ConstraintPolicy m_constraint;
};
}  // namespace rover_domain

#endif