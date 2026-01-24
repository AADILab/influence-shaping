#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_IREWARD
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_IREWARD

#include <rover_domain/core/detail/pack.hpp>

namespace rover_domain {

/*
 *
 * Reward interface for bindings
 *
 */
class IReward {
   public:
    [[nodiscard]] virtual double compute(const AgentPack&) const = 0;
    virtual ~IReward() = default;
};

}  // namespace rover_domain

#endif