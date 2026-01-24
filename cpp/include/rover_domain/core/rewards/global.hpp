#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_GLOBAL
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_GLOBAL

#include <rover_domain/core/rover/irover.hpp>
#include <rover_domain/core/poi/ipoi.hpp>

namespace rover_domain {

/*
 *
 * Default environment reward: checks if all constraints are satisfied
 *
 */
class Global {
   public:
    [[nodiscard]] double compute(const Agents& agents, const POIs& pois, int unused_idx) const {
        double reward = 0.0;
        for (int i = 0; i < pois.size(); ++i) {
            reward = reward + pois[i]->value()*pois[i]->constraint_satisfied(pois, agents, i);
        }
        return reward;
    }
    [[nodiscard]] double compute_without_me(const Agents& agents, const POIs& pois, int idx) const {
        // Build vector of agents without me
        Agents agents_without_me;
        for (int i=0; i < agents.size(); ++i) {
            if (i != idx) {
                agents_without_me.push_back(agents[i]);
            }
        }
        double reward_without_me = compute(agents_without_me, pois, 0);
        return reward_without_me;
    }
    [[nodiscard]] double compute_without_inds(const Agents& agents, const POIs& pois, std::vector<int> inds) const {
        // Build a vector of agents that excludes the specified inds
        std::vector<Agent> agents_without_inds;
        for (int i=0; i < agents.size(); ++i) {
            // Check that i is not an ind that we are removing
            if (std::find(inds.begin(), inds.end(), i) == inds.end()) {
                agents_without_inds.push_back(agents[i]);
            }
        }
        double reward_without_inds = compute(agents_without_inds, pois, 0);
        return reward_without_inds;
    }
};

}  // namespace rover_domain

#endif