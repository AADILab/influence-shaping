#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ENVIRONMENT
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ENVIRONMENT

#include <Eigen/Dense>
#include <rover_domain/core/poi/default_poi.hpp>
#include <rover_domain/core/rover/rover.hpp>
#include <rover_domain/core/sensors/lidar.hpp>
#include <rover_domain/core/rewards/computer.hpp>
#include <rover_domain/core/setup/init_corners.hpp>
#include <tuple>
#include <vector>

namespace rover_domain {

/*
 *
 * Default Rovers environment
 *
 */
template <typename InitPolicy = CornersInit>
class Environment {
   public:
    using Action = Eigen::MatrixXd;
    using State = std::vector<Eigen::MatrixXd>;
    using Reward = std::vector<double>;

    Environment(InitPolicy initPolicy = InitPolicy(), std::vector<Agent> rovers = {},
                std::vector<POI> pois = {}, size_t width = 10.0, size_t height = 10.0,
                bool debug_reward_equals_G = false)
        : m_initPolicy(initPolicy),
          m_rovers(std::move(rovers)),
          m_pois(std::move(pois)),
          m_width(width),
          m_height(height),
          m_reward_computer(m_rovers, m_pois, debug_reward_equals_G) {}

    // helpers to set rovers/pois after the fact
    void set_rovers(std::vector<Agent> rovers) { m_rovers = std::move(rovers); }
    void set_pois(std::vector<POI> pois) { m_pois = std::move(pois); }

    const std::vector<Agent>& rovers() { return m_rovers; }
    const std::vector<POI>& pois() { return m_pois; }

    void perform_step(std::vector<Action> actions) {
        for (size_t i = 0; i < m_rovers.size(); ++i) {
            auto& rover = m_rovers[i];
            // call update for all rovers
            rover->update();
            // take actions
            rover->act(actions[i]);
            // bound position
            clamp_bounds(rover);
        }
        // call update for pois
        for (auto& poi : m_pois) poi->update();
    }

    std::tuple<State, Reward> step(std::vector<Action> actions) {
        perform_step(actions);
        // return next observations and rewards
        return status();
    }

    State step_without_rewards(std::vector<Action> actions) {
        perform_step(actions);
        // Just return observations
        return observations();
    }

    State observations() {
        State state;
        for (int i = 0; i < m_rovers.size(); ++i) {
            // std::cout << "Environment::status() | i | " << i << std::endl;
            // Construct the AgentPack on the fly
            // const AgentPack pack = {i, m_rovers, m_pois};
            // std::cout << "pack" << std::endl;
            state.push_back(m_rovers[i]->scan(m_rovers, m_pois, i));
            // rewards.push_back(m_rovers[i]->reward(pack));
        }
        return state;
    }

    std::tuple<State, Reward> reset() {
        // std::cout << "Environment::reset()" << std::endl;
        // clear agents
        for (auto& r : m_rovers) r->reset();
        // reset pois
        for (auto& poi : m_pois) poi->set_observed(false);
        // initialize
        m_initPolicy.initialize(m_rovers, m_pois);
        // return next observations and rewards
        return status();
    }

    void render() {}
    void close() {}

    const size_t& width() { return m_width; }
    const size_t& height() { return m_height; }
    // TODO add pre/post update for all components

    Reward rewards() {
        return m_reward_computer.compute();
    }

    std::tuple<State, Reward> status() {
        // Give us the full status of the environment,
        // including what every agent observes, and the rewards for each agent
        return {observations(), rewards()};
    }

   private:
    inline void clamp_bounds(Agent& rover) {
        // First check for agent specific bounds
        rover->set_position(std::clamp(rover->position().x, rover->bounds().m_low_x, rover->bounds().m_high_x),
                            std::clamp(rover->position().y, rover->bounds().m_low_y, rover->bounds().m_high_y));
        // Then bound the agent by the map bounds
        rover->set_position(std::clamp(rover->position().x, 0.0, 1.0 * m_width),
                            std::clamp(rover->position().y, 0.0, 1.0 * m_height));
    }
    InitPolicy m_initPolicy;
    std::vector<Agent> m_rovers;
    std::vector<POI> m_pois;
    RewardComputer m_reward_computer;

    size_t m_width;
    size_t m_height;
};

/*
 *
 * Syntactic sugar for agents/entities
 *
 */
using Agents = std::vector<Agent>;
using POIs = std::vector<POI>;
using Actions = std::vector<Eigen::MatrixXd>;

Eigen::MatrixXd tensor(std::vector<double> list) {
    return Eigen::Map<Eigen::MatrixXd>(list.data(), list.size(), 1);
}

Agents& operator<<(Agents& vector, Agent&& rover) {
    vector.push_back(std::move(rover));
    return vector;
}
POIs& operator<<(POIs& vector, POI&& poi) {
    vector.push_back(std::move(poi));
    return vector;
}

}  // namespace rover_domain

#endif
