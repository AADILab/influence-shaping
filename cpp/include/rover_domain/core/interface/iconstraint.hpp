#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI_ICONSTRAINT
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI_ICONSTRAINT

#include <rover_domain/core/rover/rover.hpp>
#include <rover_domain/core/declare/agent_types.hpp>
#include <rover_domain/core/declare/entity_types.hpp>
#include <rover_domain/core/interface/ipoi.hpp>
#include <rover_domain/utilities/math/norms.hpp>
#include <algorithm>
#include <vector>
#include <limits>

namespace rover_domain {

/*
 *
 * Constraint interface for bindings
 *
 */
class IConstraint {
   public:
    [[nodiscard]] virtual double is_satisfied(const POIs& pois, const Agents& agents, int poi_idx) const = 0;
    virtual ~IConstraint() = default;
};

}  // namespace rover_domain

#endif