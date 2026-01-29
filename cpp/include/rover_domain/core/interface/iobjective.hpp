#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI_IOBJECTIVE
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI_IOBJECTIVE

#include <rover_domain/core/declare/agent_types.hpp>
#include <rover_domain/core/declare/poi_types.hpp>

namespace rover_domain {

/*
 *
 * Objectives interface for bindings
 *
 */
class IObjective {
   public:
    [[nodiscard]] virtual double score(const POIs& pois, const Agents& agents, int poi_idx) const = 0;
    virtual ~IObjective() = default;
};

}  // namespace rover_domain

#endif