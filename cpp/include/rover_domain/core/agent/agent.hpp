#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER

#include <rover_domain/core/interface/iagent.hpp>
#include <vector>

namespace rover_domain {

/*
 *
 * Default boilerplate rover
 *
 */
template <typename SensorType>
class Rover final : public IAgent {
    using SType = thyme::utilities::SharedWrap<SensorType>;
    using ActionType = std::vector<double>;
   public:
    Rover(Bounds bounds, IndirectDifferenceParameters indirect_difference_parameters, std::string reward_type, std::string type_, double obs_radius = 1.0, SType sensor = SensorType())
        : IAgent(bounds, indirect_difference_parameters, reward_type, type_, obs_radius), m_sensor(sensor) {}
    // NOTE: This is commented out because I couldn't get it to work properly, but left as dead code to help me later if I need to get it working
    // Rover(const Rover& rover)
    //     : IAgent(rover.indirect_difference_parameters(), rover.reward_type(), rover.type(), rover.obs_radius()), m_sensor(SensorType()), m_reward(RewardType()) {}
    [[nodiscard]] virtual std::vector<double> scan(const Agents& agents, const POIs& pois, int agent_idx) const override {
        // std::cout << "Rover::scan()" << std::endl;
        return m_sensor->scan(agents, pois, agent_idx);
    }
    void act(const ActionType& action) override {
        update_position(action[0], action[1]);
    }


   public:
    SType m_sensor;
};

/*
 *
 * Example of bringing in a new Rover from the python bindings
 *
 */
// class Drone final : public IAgent {
//    public:
//     Drone(double obs_radius = 1.0) : IAgent(obs_radius) {}

//     [[nodiscard]] virtual std::vector<double> scan(const Agents& agents, const POIs& pois, int agent_idx) const override { return {}; }
//     [[nodiscard]] virtual double reward(const Agents& agents, const POIs& pois, int agent_idx) const override { return 0; }
//     void act(const std::vector<double>&) override { }
// };

}  // namespace rover_domain

#endif
