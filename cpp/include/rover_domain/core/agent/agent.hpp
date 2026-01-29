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
class DefaultAgent final : public IAgent {
    using SType = thyme::utilities::SharedWrap<SensorType>;
    using ActionType = std::vector<double>;
   public:
    DefaultAgent(Bounds bounds, IndirectDifferenceParameters indirect_difference_parameters, std::string reward_type, std::string type_, double obs_radius = 1.0, SType sensor = SensorType())
        : IAgent(bounds, indirect_difference_parameters, reward_type, type_, obs_radius), m_sensor(sensor) {}
    [[nodiscard]] virtual std::vector<double> scan(const Agents& agents, const POIs& pois, int agent_idx) const override {
        return m_sensor->scan(agents, pois, agent_idx);
    }
    void act(const ActionType& action) override {
        update_position(action[0], action[1]);
    }


   public:
    SType m_sensor;
};

}  // namespace rover_domain

#endif
