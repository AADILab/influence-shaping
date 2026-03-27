#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_REWARD_COMPUTER
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_REWARD_COMPUTER

#include <rover_domain/core/agent/agent.hpp>
#include <rover_domain/utilities/math/norms.hpp>
#include <rover_domain/core/sensors/lidar.hpp>

namespace rover_domain {

using CompleteInfluenceArray = std::vector<std::vector<std::vector<bool>>>;
using SliceInfluenceArray = std::vector<std::vector<bool>>;

class RewardComputer {
    public:
    using Reward = std::vector<double>;

    RewardComputer(std::vector<Agent> rovers, std::vector<POI> pois, bool debug_reward_equals_G) {
        // std::cout << "RewardComputer::RewardComputer()" << std::endl;
        m_rovers = rovers;
        m_pois = pois;
        m_debug_reward_equals_G = debug_reward_equals_G;
    }

    /* Create a complete influence array indexed by [t][i][k] where
    t is the timestep, i is the agent exerting influence, and k is the agent being influenced
    Values are 0 or 1, indicating that between two agents there either is influence or there is not influence
    */
    std::vector<std::vector<std::vector<bool>>> create_complete_influence_array() const {
        // Figure out how many timesteps are in the paths
        int t_final = m_rovers[0]->path().size();

        // initialize this array with all zeros
        std::vector<std::vector<std::vector<bool>>> complete_influence_array(
            t_final, std::vector<std::vector<bool>>(
                m_rovers.size(), std::vector<bool>(m_rovers.size(), 0)
            )
        );

        // populate the 1s where agents are influenced
        for (int t=0; t < t_final; ++t) { // outer loop is time
            for (int i=0; i < m_rovers.size(); ++i) { // middle loop is agent that is exerting influence
                for (int k=0; k < m_rovers.size(); ++k) { // inner loop is agent that is being influenced
                    // Check that agent i is influencing agent k, and set the influence value to 1 if this is the case
                    if (
                        i != k && m_rovers[i]->type() == AgentType::UAV && m_rovers[k]->type() == AgentType::Rover && is_influencing(m_rovers[i], m_rovers[k], t)
                    ) {
                        complete_influence_array[t][i][k] = 1;
                    }
                }
            }
        }

        return complete_influence_array;
    }

    /* Create a local influence array that just tells us who agent i influenced at different times
    Indexing is [t][k] where t is the timestep, and k is the agent being influenced
    0 means this agent was not influenced by agent i, and 1 means this agent was influenced by agent i*/
    std::vector<std::vector<bool>> create_local_influence_array(
        std::vector<std::vector<std::vector<bool>>> complete_influence_array,
        int i
    ) const {
        // Figure out how many timesteps are in the paths
        int t_final = m_rovers[0]->path().size();

        // Give us an empty array to start
        std::vector<std::vector<bool>> local_influence_array(
            t_final, std::vector<bool>(
                m_rovers.size(), 0
            )
        );

        // Now populate based on the complete influence array
        for (int t=0; t < t_final; ++t) { // Iterate through time
            for (int k=0; k < m_rovers.size(); ++k) { // Iterate through agents that are being influenced
                if (complete_influence_array[t][i][k] == 1) {
                    local_influence_array[t][k] = 1;
                }
            }
        }
        return local_influence_array;
    }

    /* Create an all or nothing influence array that tells us who agent i influenced based on who influenced who the most
    Winner takes all here, so if two agents influenced the same agent at the same time, only one gets the credit.
    When thinking in terms of timesteps, that means we're using this to resolve ties
    Indexing is [t][k] where t is the timestep, k is the agent being influenced*/
    std::vector<std::vector<bool>> create_allornothing_influence_array(
        std::vector<std::vector<std::vector<bool>>> complete_influence_array,
        int i
    ) const {
        // Figure out how many timesteps are in the paths
        int t_final = m_rovers[0]->path().size();

        // Give us an empty array to start
        std::vector<std::vector<bool>> allornothing_influence_array(
            t_final, std::vector<bool>(
                m_rovers.size(), 0
            )
        );

        // Now populate based on the complete influence array
        for (int t=0; t < t_final; ++t) { // Iterate through time
            for (int k=0; k < m_rovers.size(); ++k) { // Iterate through agents that are being influenced
                // Iterate through agents that are exerting influence and only give this agent credit if it is the
                // leftmost agent to exert an influence.
                // (Yes, this is overly complicated for now, but this infrastructure will be helpful when this becomes more complicated)
                int i_credit = -1;
                int highest_influence = -1;
                for (int i_=0; i_ < m_rovers.size(); ++i_) {
                    if (complete_influence_array[t][i_][k] > highest_influence) {
                        i_credit = i_;
                        highest_influence = complete_influence_array[t][i_][k];
                    }
                }
                if (i_credit == i) {
                    allornothing_influence_array[t][k] = 1;
                }


                // if (complete_influence_array[t][0][k] == 1 && i == 0) {
                //     allornothing_influence_array[t][k] = 1;
                // }
                // else {
                //     bool resolved = false;
                //     bool found = false;
                //     int i_ = 0;
                //     while (!resolved) {
                //         i_++;
                //         if ( complete_influence_array[t][i_-1][k] == 0 && complete_influence_array[t][i_][k] == 1) {
                //             found = true;
                //             resolved = true;
                //         }
                //         if (i_ >= m_rovers.size()-1) {
                //             resolved = true;
                //         }
                //     }
                //     if (found && i_ == i) {
                //         allornothing_influence_array[t][k] = 1;
                //     }
                // }
            }
        }

        // std::cout << "allornothing_influence_array for agent i : " << i << std::endl;
        // for (int t=0; t<t_final; ++t) {
        //     for (int k=0; k<m_rovers.size(); ++k) {
        //         std::cout << "[t][k] : value " << "["<<t<<"]["<<k<<"] : "<<allornothing_influence_array[t][k] << std::endl;
        //     }
        // }

        return allornothing_influence_array;
    }

    /* Create system influence array that tells us when agent k was influenced
    by any agent in the system at a particular timestep
    Indexing is [t][k] where t is the timestep and k is the agent being influenced
    OPTIONAL: if agent i_ is specified, then we will not consider agent i_'s influence
    as part of the system when constructing the system influence array
    */
    std::vector<std::vector<bool>> create_system_influence_array(
        std::vector<std::vector<std::vector<bool>>> complete_influence_array,
        int i_ = -1
    ) const {
        // Figure out how many timesteps in the path
        int t_final = m_rovers[0]->path().size();

        // Start with empty array
        std::vector<std::vector<bool>> system_influence_array(
            t_final, std::vector<bool>(
                m_rovers.size(), 0
            )
        );

        // Populate the array
        for (int t=0; t < t_final; ++t) { // Iterate through time
            for (int k=0; k < m_rovers.size(); ++k) { // Iterate through agents that were influenced this step
                // If this agent was actually influenced this step, put a 1 for system influence. Else, leave it as 0
                bool k_was_influenced = false;
                for (int i=0; i < m_rovers.size(); ++i) {
                    if (complete_influence_array[t][i][k] == 1 && (i_ == -1 || i_ != i)) {
                        k_was_influenced = true;
                    }
                }
                if (k_was_influenced) {
                    system_influence_array[t][k] = 1;
                }
            }
        }
        return system_influence_array;
    }

    /* Create difference influence array that gives us the difference between two input arrays
    We only get a 1 for influence if arr_x is 1 and arr_y is 0
    Indexing is [t][k] where t is the timestep and k is the agent being influenced
    */
    std::vector<std::vector<bool>> create_difference_influence_array(
        std::vector<std::vector<bool>> arr_x,
        std::vector<std::vector<bool>> arr_y
    ) const {
        // Get timesteps
        int t_final = m_rovers[0]->path().size();

        // Initialize this array with all zeros
        std::vector<std::vector<bool>> difference_influence_array(
            t_final, std::vector<bool>(
                m_rovers.size(), 0
            )
        );

        // Now populate based on input influence arrays
        for (int t=0; t < t_final; ++t) { // Iterate through time
            for (int k=0; k < m_rovers.size(); ++k) { // Iterate through agents being influenced
                if (arr_x[t][k] == 1 && arr_y[t][k] == 0) {
                    difference_influence_array[t][k] = 1;
                }
            }
        }
        return difference_influence_array;
    }

    /* Create a set of agents with paths that place that agent at [-1, -1] if that agent was influenced according to the input
    influence array */
    std::vector<Agent> create_counterfactual_rovers(std::vector<Agent> rovers, std::vector<std::vector<bool>> influence_array) const {
        // Figure out how many timesteps are in the paths
        int t_final = m_rovers[0]->path().size();

        // empty vector of counterfactual rovers
        std::vector<Agent> counterfactual_rovers;

        // Populate counterfactual rovers with copies of the rovers
        // Clear the path of each one
        for (int k=0; k < rovers.size(); ++k) {
            DefaultAgent<Lidar<Density>> rover(
                rovers[k]->bounds(),
                rovers[k]->reward_spec(),
                rovers[k]->type(),
                rovers[k]->obs_radius()
            );
            rover.reset();
            counterfactual_rovers.push_back(rover);
        }

        // Now populate the paths, but use the influence array to counterfactually put the position at [-1, -1] if that agent was influenced
        for (int t=0; t < t_final; ++t) {
            for (int k=0; k < m_rovers.size(); ++k) {
                if (influence_array[t][k] == 1) {
                    counterfactual_rovers[k]->set_position(-1, -1);
                }
                else {
                    counterfactual_rovers[k]->set_position(
                        m_rovers[k]->path()[t].x,
                        m_rovers[k]->path()[t].y
                    );
                }
            }
        }

        // Give us the rovers with counterfactual paths
        return counterfactual_rovers;
    }

    std::vector<std::vector<int>> prep_all_or_nothing_influence() const {
        // Each element contains the indicies of rovers (as in, nominal type "rover") influenced
        // by the agent in this index.
        // (Only going to count nominal type "uav" agents as being able to influence)

        // std::cout << "RewardComputer::prep_all_or_nothing_influence()" << std::endl;

        int t_final = m_rovers[0]->path().size();

        // std::cout << "RewardComputer::prep_all_or_nothing_influence() t_final | " << t_final << std::endl;

        // Counters tell us how much each agent was influenced by other agents
        // First index (k) is the agent being influenced
        // Second index (i) is how much agent i influenced agent k
        std::vector<std::vector<int>> counters(m_rovers.size(), std::vector<int>(m_rovers.size(), 0));
        for (int t=0; t < t_final; ++t) {
            // std::cout << "t " << t << std::endl;
            // agent i is the influencing agent
            for (int i=0; i < m_rovers.size(); ++i) {
                // std::cout << "i " << i << std::endl;
                // agent k is the agent being influenced
                for (int k=0; k < m_rovers.size(); ++k) {
                    // std::cout << "k " << k << std::endl;
                    if (i != k && m_rovers[i]->type() == AgentType::UAV && m_rovers[k]->type() == AgentType::Rover && is_influencing(m_rovers[i], m_rovers[k], t) ) {
                        // std::cout << "Increasing counter at counters["<<k<<"]["<<i<<"]" << std::endl;
                        counters[k][i]++;
                    }
                }
            }
        }

        // std::cout << "RewardComputer::prep_all_or_nothing_influence() Finished creating counters" << std::endl;
        // std::cout << "counters.size() " << counters.size() << std::endl;
        for (int k=0; k < counters.size(); ++k) {
            // std::cout << "counters[" << k << "]" << std::endl;
            for (int i=0; i < counters.size(); ++i) {
                // std::cout << "counters[" << k << "][" << i << "] = " << counters[k][i] << std::endl;
            }
        }
        // std::cout << "counters " << counters[0] << std::endl;

        // Create the sets of agents to remove for each agent
        // Index is the agent being influenced (k)
        // This index gives a vector of indicies of agents that agent k influenced
        std::vector<std::vector<int>> influence_sets(m_rovers.size(), std::vector<int>({}));
        // Include yourself in your influence set
        // std::cout << "RewardComputer::prep_all_or_nothing_influence() Insert yourself into your influence set (start)" << std::endl;
        for (int i=0; i < influence_sets.size(); ++i) {
            influence_sets[i].push_back(i);
            // std::cout << "RewardComputer::prep_all_or_nothing_influence() Ran influence_sets[i].push_back(i) with i = " << i << std::endl;
        }
        // std::cout << "RewardComputer::prep_all_or_nothing_influence() Adding other agents to influence sets" << std::endl;
        for (int k=0; k < m_rovers.size(); ++k) {
            // std::cout << "RewardComputer::prep_all_or_nothing_influence() on agent k = " << k << std::endl;
            int highest_ind = -1;
            int num_influence = 0;

            // std::cout << "Beginning iteration through counters[" << k << "].size()" << std::endl;
            for (int i=0; i < counters[k].size(); ++i) {
                // std::cout << "RewardComputer::prep_all_or_nothing_influence() k = " << k << " , i = " << i << std::endl;
                if (counters[k][i] > num_influence) {
                    num_influence = counters[k][i];
                    highest_ind = i;
                }
            }

            // Who was agent k most influenced by?
            // Agent i gets credit for influencing agent k (unless agent i == -1, meaning there was no agent that influenced agent k)
            if (highest_ind != -1) {influence_sets[highest_ind].push_back(k);}
            // influence_sets[highest_ind].push_back(k);
            // std::cout << "RewardComputer::prep_all_or_nothing_influence() Ran influence_sets[highest_ind].push_back(k) on k = " << k << std::endl;
        }

        // std::cout << "RewardComputer::prep_all_or_nothing_influence() Finished building influence_sets" << std::endl;

        return influence_sets;
    }

    // TODO: This is based on position RIGHT NOW of each agent
    // need to make this based on position of agents at A PARTICULAR POINT IN TIME ALONG THEIR PATHS
    int is_influencing(Agent agent0, Agent agent1, int t) const {
        if (l2_norm(agent0->path()[t], agent1->path()[t]) <= 5.0) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }

    double reward_from_influence_array(int i, double G, const std::vector<std::vector<bool>>& influence_array) const {
        std::vector<Agent> counterfactual_rovers = create_counterfactual_rovers(m_rovers, influence_array);
        return G - global_without_inds(counterfactual_rovers, m_pois, std::vector<int>{i});
    }

    std::vector<std::vector<bool>> make_dynamic_influence_array(int i, IDDynamic::Credit credit) const {
        const CompleteInfluenceArray complete = create_complete_influence_array();
        switch (credit) {
            case IDDynamic::Credit::Local:
                return create_local_influence_array(complete, i);
            case IDDynamic::Credit::WinnerTakesAll:
                return create_allornothing_influence_array(complete, i);
            case IDDynamic::Credit::System:
                return create_system_influence_array(complete);
            case IDDynamic::Credit::Difference: {
                const SliceInfluenceArray system = create_system_influence_array(complete);
                const SliceInfluenceArray system_without_i = create_system_influence_array(complete, i);
                return create_difference_influence_array(system, system_without_i);
            }
        }
        throw std::runtime_error("Unhandled IDDynamic::Credit");
    }

    double compute_indirect_reward(
        int i,
        double G,
        const std::vector<std::vector<int>>& influence_sets,
        const IndirectDifferenceReward& indirect
    ) const {
        double reward = 0.0;

        // First level: check which mode (IDStatic, IDDynamic, IDAdaptive)
        if (std::holds_alternative<IDStatic>(indirect.params)) {
            const IDStatic& static_mode = std::get<IDStatic>(indirect.params);

            // Second level: within IDStatic, check which type (IDStaticManual, IDStaticAutomatic)
            if (std::holds_alternative<IDStaticManual>(static_mode)) {
                const IDStaticManual& manual = std::get<IDStaticManual>(static_mode);
                reward = G - global_without_inds(m_rovers, m_pois, manual.manual);
            } else if (std::holds_alternative<IDStaticAutomatic>(static_mode)) {
                const IDStaticAutomatic& automatic = std::get<IDStaticAutomatic>(static_mode);
                reward = G - global_without_inds(m_rovers, m_pois, influence_sets[i]);
            } else {
                throw std::runtime_error("Unhandled IDStatic variant");
            }
        }
        else if (std::holds_alternative<IDDynamic>(indirect.params)) {
            const IDDynamic& dynamic = std::get<IDDynamic>(indirect.params);
            const SliceInfluenceArray influence_array = make_dynamic_influence_array(i, dynamic.credit);
            reward = reward_from_influence_array(i, G, influence_array);
        }
        else if (std::holds_alternative<IDAdaptive>(indirect.params)) {
            throw std::runtime_error("IDAdaptive is not implemented yet");
        }
        else {
            throw std::runtime_error("Unhandled indirect mode variant");
        }

        if (indirect.add_G) {
            reward += G;
        }
        return reward;
            return reward;
    }

    double compute_reward_for_agent(
        int i,
        double G,
        const std::vector<std::vector<int>>& influence_sets
    ) const {
        const RewardSpec reward_spec = m_rovers[i]->reward_spec();
        if (std::holds_alternative<GlobalReward>(reward_spec)) {
            return G;
        }
        else if (std::holds_alternative<DifferenceReward>(reward_spec)) {
            return G - global_without_me(m_rovers, m_pois, i);
        }
        else if (std::holds_alternative<IndirectDifferenceReward>(reward_spec)) {
            const IndirectDifferenceReward& indirect = std::get<IndirectDifferenceReward>(reward_spec);
            return compute_indirect_reward(i, G, influence_sets, indirect);
        }
        else {
            throw std::runtime_error("Unhandled RewardSpec variant");
        }
    }

    [[nodiscard]] Reward compute() const {
        // std::cout << "Reward::compute()" << std::endl;
        Reward rewards;
        // Compute G
        double G = global(m_rovers, m_pois);
        // std::cout << "Reward::compute() Computed G" << std::endl;
        // Prep for computing Indirect D
        std::vector<std::vector<int>> influence_sets = prep_all_or_nothing_influence();
        // std::cout << "Reward::compute() Computed influence_sets" << std::endl;

        // Now compute the rewards for each agent
        // std::cout << "Reward::compute() Computing rewards for each agent" << std::endl;
        for (int i = 0; i < m_rovers.size(); ++i) {
            // std::cout << "Reward::compute() Computing reward for agent " << i << std::endl;
            double reward = compute_reward_for_agent(i, G, influence_sets);
            if (m_debug_reward_equals_G && reward != G) {
                throw std::runtime_error("reward does not equal G!");
            }
            rewards.push_back(reward);
        }
        return rewards;
    }

    [[nodiscard]] double global(const Agents& agents, const POIs& pois) const {
        double reward = 0.0;
        for (int i = 0; i < pois.size(); ++i) {
            reward = reward + pois[i]->value()*pois[i]->score(pois, agents, i);
        }
        return reward;
    }
    [[nodiscard]] double global_without_me(const Agents& agents, const POIs& pois, int agent_idx) const {
        // Build vector of agents without me
        std::vector<Agent> agents_without_me;
        for (int i=0; i < agents.size(); ++i) {
            if (i != agent_idx) {
                agents_without_me.push_back(agents[i]);
            }
        }
        return global(agents_without_me, pois);
    }

    [[nodiscard]] double global_without_inds(const Agents& agents, const POIs& pois, std::vector<int> inds) const {
        // std::cout << "Reward::compute_without_inds()" << std::endl;
        // Build a vector of agents that excludes the specified inds
        std::vector<Agent> agents_without_inds;
        for (int i=0; i < agents.size(); ++i) {
            // Check that i is not an ind that we are removing
            if (std::find(inds.begin(), inds.end(), i) == inds.end()) {
                agents_without_inds.push_back(agents[i]);
            }
        }
        return global(agents_without_inds, pois);
    }

    bool get_debug_reward_equals_G() {
        return m_debug_reward_equals_G;
    }

    std::vector<Agent> m_rovers;
    std::vector<POI> m_pois;

    private:
    bool m_debug_reward_equals_G; // private so you can't change it after the class has been initialized
};

}  // namespace rover_domain

#endif
