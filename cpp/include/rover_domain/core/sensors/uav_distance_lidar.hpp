#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_UAV_DISTANCE_LIDAR
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_UAV_DISTANCE_LIDAR

#include <rover_domain/core/interface/iagent.hpp>
#include <rover_domain/core/interface/isensor.hpp>
#include <rover_domain/core/composition/density.hpp>
#include <rover_domain/utilities/math/norms.hpp>
#include <vector>

namespace rover_domain {

/*
 *
 * UavDistanceLidar - returns distance to each UAV in the environment
 *
 */
class UavDistanceLidar : public ISensor {
   public:
    // Default constructor
    UavDistanceLidar()
        : m_num_sensed_uavs(0) {}

    [[nodiscard]] std::vector<double> scan(const Agents& agents, const POIs& pois, int agent_idx) const {
        auto& agent = agents[agent_idx];
        m_num_sensed_uavs = 0;
        std::vector<double> distances;

        // Iterate through all agents
        for (std::size_t i = 0; i < agents.size(); ++i) {
            // Only process UAVs
            if (agents[i]->type() == AgentType::UAV) {
                auto& sensed_agent = agents[i];
                double distance = thyme::math::l2_norm(agent->position(), sensed_agent->position());

                if (distance <= agent->obs_radius()) {
                    distances.push_back(distance / agent->obs_radius());
                    m_num_sensed_uavs++;
                } else {
                    distances.push_back(-1.0);
                }
            }
        }

        return distances;
    }

    [[nodiscard]] inline int num_sensed_uavs() const {
        return m_num_sensed_uavs;
    }

   private:
    mutable int m_num_sensed_uavs;
};

}  // namespace rover_domain

#endif
