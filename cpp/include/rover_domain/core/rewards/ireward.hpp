#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_IREWARD
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_IREWARD

#include <rover_domain/core/detail/agent_types.hpp>
#include <rover_domain/core/detail/entity_types.hpp>

namespace rover_domain {

/*
 *
 * Reward interface for bindings
 *
 */
class IReward {
   public:
    [[nodiscard]] virtual double compute(const Agents& agents, const POIs& pois, int unused_idx) const = 0;
    virtual ~IReward() = default;
};

}  // namespace rover_domain

#endif