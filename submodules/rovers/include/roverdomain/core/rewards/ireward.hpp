#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_IREWARD
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_IREWARD

#include <roverdomain/core/detail/pack.hpp>

namespace rovers::rewards {

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

}  // namespace rovers::rewards

#endif