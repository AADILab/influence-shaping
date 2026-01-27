#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER

#include <Eigen/Dense>
#include <iostream>
#include <rover_domain/core/declare/agent_types.hpp>
#include <rover_domain/core/declare/entity_types.hpp>
#include <rover_domain/core/rewards/global.hpp>
#include <rover_domain/utilities/math/cartesian.hpp>
#include <vector>
#include <rover_domain/core/rover/irover.hpp>

namespace rover_domain {

/*
 *
 * Default boilerplate rover
 *
 */
template <typename SensorType, typename ActionSpace, typename RewardType = Global>
class Rover final : public IAgent {
    using SType = thyme::utilities::SharedWrap<SensorType>;
    using RType = thyme::utilities::SharedWrap<RewardType>;
    using ActionType = Eigen::MatrixXd;
   public:
    Rover(Bounds bounds, IndirectDifferenceParameters indirect_difference_parameters, std::string reward_type, std::string type_, double obs_radius = 1.0, SType sensor = SensorType(), RType reward = RewardType())
        : IAgent(bounds, indirect_difference_parameters, reward_type, type_, obs_radius), m_sensor(sensor), m_reward(reward) {}
    // NOTE: This is commented out because I couldn't get it to work properly, but left as dead code to help me later if I need to get it working
    // Rover(const Rover& rover)
    //     : IAgent(rover.indirect_difference_parameters(), rover.reward_type(), rover.type(), rover.obs_radius()), m_sensor(SensorType()), m_reward(RewardType()) {}
    [[nodiscard]] virtual Eigen::MatrixXd scan(const Agents& agents, const POIs& pois, int agent_idx) const override {
        // std::cout << "Rover::scan()" << std::endl;
        return m_sensor->scan(agents, pois, agent_idx);
    }
    // TODO: Replace this function with an enum that describes which reward to give this agent (Global, Difference, etc)
    [[nodiscard]] virtual double reward(const Agents& agents, const POIs& pois, int agent_idx) const override {
        // each aget gets a reward set here but only nominally so the reward computer knows
        // what to do
        // but each agent is not comjputing its own reward
        // std::cout << "Rover::reward()" << std::endl;
        return m_reward->compute(agents, pois);
    }
    void act(const ActionType& action) override {
        // default, move in x and y
        assert(action.rows() >= 2);
        auto act = static_cast<Eigen::Vector2d>(action);
        update_position(act[0], act[1]);
    }


   public:
    SType m_sensor;
    RType m_reward;
};

/*
 *
 * Example of bringing in a new Rover from the python bindings
 *
 */
// class Drone final : public IAgent {
//    public:
//     Drone(double obs_radius = 1.0) : IAgent(obs_radius) {}

//     [[nodiscard]] virtual Eigen::MatrixXd scan(const Agents& agents, const POIs& pois, int agent_idx) const override { return {}; }
//     [[nodiscard]] virtual double reward(const Agents& agents, const POIs& pois, int agent_idx) const override { return 0; }
//     void act(const Eigen::MatrixXd&) override { }
// };

}  // namespace rover_domain

#endif
