#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_DIFFERENCE
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_DIFFERENCE

#include <rover_domain/core/rewards/global.hpp>
#include <rover_domain/utilities/ranges.hpp>

namespace rover_domain {

/*
 *
 * Difference between reward and the reward without the agent acting
 *
 */
class Difference {
   public:
    [[nodiscard]] double compute(const AgentPack& pack) const {
        // std::cout << "Difference::compute()" << std::endl;
        double reward = Global().compute(pack);
        // Make a vector of agents with the appropriate agent removed
        std::vector<Agent> agents_without_me;
        for (int i = 0; i < pack.agents.size(); ++i) {
            if (i != pack.agent_index) {
                agents_without_me.push_back(pack.agents[i]);
            }
        }
        // std::cout << "Difference::compute() | Built pack_without_me" << std::endl;
        // Make a new agentpack. Use dummy variable for agent index.
        const AgentPack& pack_without_me = AgentPack(0, agents_without_me, pack.entities);
        double reward_without_me = Global().compute(pack_without_me);
        return reward - reward_without_me;
    }
};

// class IndirectDifference {
//     public:
//     [[nodiscard]] double compute(const AgentPack& pack) const {
//         double reward = Global().compute(pack)
//         // Figure out
//         return reward;
//     }
// };

}  // namespace rover_domain

#endif