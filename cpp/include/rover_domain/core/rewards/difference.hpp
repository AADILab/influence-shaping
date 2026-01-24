#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_DIFFERENCE
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_DIFFERENCE

#include <rover_domain/core/rewards/global.hpp>
#include <functional>
#include <rover_domain/core/detail/agent_types.hpp>
#include <rover_domain/core/detail/entity_types.hpp>
#include <rover_domain/core/rover/irover.hpp>
#include <rover_domain/core/poi/ipoi.hpp>

namespace rover_domain {

/*
 *
 * Difference between reward and the reward without the agent acting
 *
 */
class Difference {
   public:
    [[nodiscard]] double compute(const Agents& agents, const POIs& pois, int idx) const {
        // std::cout << "Difference::compute()" << std::endl;
        double reward = Global().compute(agents, pois, 0);
        // Make a vector of agents with the appropriate agent removed
        std::vector<Agent> agents_without_me;
        for (int i = 0; i < agents.size(); ++i) {
            if (i != idx) {
                agents_without_me.push_back(agents[i]);
            }
        }
        double reward_without_me = Global().compute(agents_without_me, pois, 0);
        return reward - reward_without_me;
    }
};

}  // namespace rover_domain

#endif