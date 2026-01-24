#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_REWARD_COMPUTER
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_REWARD_COMPUTER

#include <rover_domain/core/rewards/global.hpp>
#include <rover_domain/core/rewards/difference.hpp>
#include <rover_domain/core/rover/rover.hpp>
#include <rover_domain/utilities/math/norms.hpp>
#include <rover_domain/core/sensors/lidar.hpp>

namespace rover_domain {

class RewardComputer {
    public:
    using Reward = std::vector<double>;

    RewardComputer(std::vector<Agent> rovers, std::vector<Entity> pois, bool debug_reward_equals_G) {
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
                        i != k && m_rovers[i]->type() == "uav" && m_rovers[k]->type() == "rover" && is_influencing(m_rovers[i], m_rovers[k], t)
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
            Rover<Lidar<Density>, Global> rover(
                rovers[k]->bounds(),
                rovers[k]->indirect_difference_parameters(),
                rovers[k]->reward_type(),
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
                    if (i != k && m_rovers[i]->type() == "uav" && m_rovers[k]->type() == "rover" && is_influencing(m_rovers[i], m_rovers[k], t) ) {
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

    [[nodiscard]] Reward compute() const {
        // std::cout << "Reward::compute()" << std::endl;
        Reward rewards;
        // Compute G
        double G = m_Global.compute(AgentPack(0, m_rovers, m_pois));
        // std::cout << "Reward::compute() Computed G" << std::endl;
        // Prep for computing Indirect D
        std::vector<std::vector<int>> influence_sets = prep_all_or_nothing_influence();
        // std::cout << "Reward::compute() Computed influence_sets" << std::endl;

        // Now compute the rewards for each agent
        // std::cout << "Reward::compute() Computing rewards for each agent" << std::endl;
        for (int i = 0; i < m_rovers.size(); ++i) {
            // std::cout << "Reward::compute() Computing reward for agent " << i << std::endl;
            double reward = 0.0;
            // Get the reward type
            std::string reward_type = m_rovers[i]->reward_type();
            // Compute the reward for this agent based on the reward type
            if (reward_type == "Global") {
                // std::cout << "Reward::compute() Computing Global reward" << std::endl;
                reward = G;
            }
            else if (reward_type == "Difference") {
                // std::cout << "Reward::compute() Computing Difference reward" << std::endl;
                reward = G - m_Global.compute_without_me(AgentPack(0, m_rovers, m_pois), i);
            }
            else if (reward_type == "IndirectDifference") {
                // std::cout << "Reward::compute() Computing Indirect Difference" << std::endl;
                // Start simple.
                // Assume that only rovers can count as being influenced
                // Use all or nothing influence assignment. Just remove the entire trajectories.
                // Refactor for more options later.
                if (m_rovers[i]->indirect_difference_parameters().m_assignment == "manual") {
                    reward = G - m_Global.compute_without_inds(AgentPack(0, m_rovers, m_pois), m_rovers[i]->indirect_difference_parameters().m_manual);
                }
                else if (m_rovers[i]->indirect_difference_parameters().m_assignment == "automatic") {
                    // Timestep based removal
                    if (m_rovers[i]->indirect_difference_parameters().m_automatic_parameters.m_timescale == "timestep") {
                        // In this route, create sets at each time step. If someone was influenced, then put a stand-in for their state as
                        // a counterfactual. For instance (need to check if this will work), put -1,-1 as the position
                        // (or if that doesn't work, add a std::vector<boolean> that has 0 for removed at step i vs 1 for present at step i. Modify G to check this bool)

                        // Construct the influence array telling us who to remove when using the specified method
                        // std::vector<std::vector<bool>> influence_array = create_allornothing_influence_array(
                        //         create_complete_influence_array(), i
                        //     )
                        std::vector<std::vector<bool>> influence_array;
                        if (m_rovers[i]->indirect_difference_parameters().m_automatic_parameters.m_credit == "Local") {
                            influence_array = create_local_influence_array(
                                create_complete_influence_array(), i
                            );
                        }
                        else if (m_rovers[i]->indirect_difference_parameters().m_automatic_parameters.m_credit == "AllOrNothing") {
                            influence_array = create_allornothing_influence_array(
                                create_complete_influence_array(), i
                            );
                        }
                        else if (m_rovers[i]->indirect_difference_parameters().m_automatic_parameters.m_credit == "System") {
                            influence_array = create_system_influence_array(
                                create_complete_influence_array()
                            );
                        }
                        else if (m_rovers[i]->indirect_difference_parameters().m_automatic_parameters.m_credit == "Difference") {
                            // Start with the complete influence array
                            std::vector<std::vector<std::vector<bool>>> complete_influence_array = create_complete_influence_array();
                            // Figure out the system influence array
                            std::vector<std::vector<bool>> system_influence_array = create_system_influence_array(complete_influence_array);
                            // Now with agent i's influence removed
                            std::vector<std::vector<bool>> counterfactual_system_influence_array = create_system_influence_array(complete_influence_array, i);
                            // The difference between system influence with i vs system influence without i is the difference influence we want
                            influence_array = create_difference_influence_array(system_influence_array, counterfactual_system_influence_array);
                        }

                        // Construct a set of counterfactual agents that have paths where that agent is at [-1, -1] if it was influenced by another agent
                        std::vector<Agent> counterfactual_rovers = create_counterfactual_rovers(
                            m_rovers, influence_array
                        );

                        // I guess this is where I can build a trace array? Or modify influence_array to contain
                        // information about the traces?
                        // I think I'd be using the same counterfactual removal
                        // except now we give a partial reward for the traces
                        // maybe
                        // m_Global.compute_with_traces()
                        // remove counterfactual rovers, but provide a partial reward based on the trace
                        // and remove yourself

                        // Now compute d-indirect using these rovers
                        // reward = G - m_Global.compute(AgentPack(0, counterfactual_rovers, m_pois));
                        // Now compute d-indirect using these rovers.
                        // Make sure to entirely remove the agent we are computing d-indirect for
                        reward = G - m_Global.compute_without_inds(AgentPack(0, counterfactual_rovers, m_pois), std::vector<int>(1, i));
                        // std::cout << "reward : " << reward << " for agent i : " << i << std::endl;
                    }

                    // Trajectory based removal
                    else if (m_rovers[i]->indirect_difference_parameters().m_automatic_parameters.m_timescale == "trajectory") {
                        // In this route, just tally it all up into one big influence set for each agent, and do the removal
                        reward = G - m_Global.compute_without_inds(AgentPack(0, m_rovers, m_pois), influence_sets[i]);
                    }
                }
            }
            if (m_rovers[i]->indirect_difference_parameters().m_add_G) {
                reward = reward + G;
            }
            if (m_debug_reward_equals_G && reward != G) {
                throw std::runtime_error("reward does not equal G!");
            }
            rewards.push_back(reward);
        }
        return rewards;
    }

    bool get_debug_reward_equals_G() {
        return m_debug_reward_equals_G;
    }

    Global m_Global;
    std::vector<Agent> m_rovers;
    std::vector<Entity> m_pois;

    private:
    bool m_debug_reward_equals_G; // private so you can't change it after the class has been initialized
};

}  // namespace rover_domain

#endif
