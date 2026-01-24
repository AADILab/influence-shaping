#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_GLOBAL
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_GLOBAL

#include <rover_domain/core/detail/pack.hpp>
#include <rover_domain/core/poi/ipoi.hpp>

namespace rovers {

/*
 *
 * Default environment reward: checks if all constraints are satisfied
 *
 */
class Global {
   public:
    [[nodiscard]] double compute(const AgentPack& pack) const {
        // std::cout << "Global::compute()" << std::endl;
        // TODO pass in a view of POIContainer filtered by observed()
        // TODO Keep filtering over this view for speed-up
        double reward = 0.0;
        for (int i = 0; i < pack.entities.size(); ++i) {
            reward = reward + pack.entities[i]->value()*pack.entities[i]->constraint_satisfied({pack.entities[i], pack.agents, pack.entities});
        }
        // for (const auto& poi : pack.entities) {
        //     // if (poi->observed()) continue;
        //     reward = reward + poi->value()*poi->constraint_satisfied({poi, pack.agents, pack.entities});
        // }
        // std::cout << "Global::compute() | Finished poi iteration" << std::endl;
        // reset pois
        // for (const auto& poi : pack.entities) poi->set_observed(false);
        return reward;
    }
    [[nodiscard]] double compute_without_me(const AgentPack& pack, int idx) const {
        // std::cout << "Reward::compute_without_me()" << std::endl;
        // Build vector of agents without me
        std::vector<Agent> agents_without_me;
        for (int i=0; i < pack.agents.size(); ++i) {
            if (i != idx) {
                agents_without_me.push_back(pack.agents[i]);
            }
        }
        const AgentPack& pack_without_me = AgentPack(0, agents_without_me, pack.entities);
        double reward_without_me = compute(pack_without_me);
        return reward_without_me;
    }
    [[nodiscard]] double compute_without_inds(const AgentPack& pack, std::vector<int> inds) const {
        // std::cout << "Reward::compute_without_inds()" << std::endl;
        // Build a vector of agents that excludes the specified inds
        std::vector<Agent> agents_without_inds;
        for (int i=0; i < pack.agents.size(); ++i) {
            // Check that i is not an ind that we are removing
            if (std::find(inds.begin(), inds.end(), i) == inds.end()) {
                agents_without_inds.push_back(pack.agents[i]);
            }
        }
        const AgentPack& pack_without_inds = AgentPack(0, agents_without_inds, pack.entities);
        double reward_without_inds = compute(pack_without_inds);
        return reward_without_inds;
    }
};

}  // namespace rovers

#endif