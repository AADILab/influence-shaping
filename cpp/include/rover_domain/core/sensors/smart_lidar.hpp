#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_SMART_LIDAR
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_SMART_LIDAR

#include <rover_domain/core/interface/iagent.hpp>
#include <rover_domain/core/interface/isensor.hpp>
#include <rover_domain/core/composition/density.hpp>
#include <vector>

namespace rover_domain {

/*
 *
 * SmartLidar - differentiates between rovers, UAVs, and POIs
 *
 */
template <typename CompositionPolicy = Density>
class SmartLidar : public ISensor {
    using CPolicy = thyme::utilities::SharedWrap<CompositionPolicy>;

   public:
    // Default constructor
    SmartLidar()
        : m_resolution(90),
          m_composition(CompositionPolicy()),
          m_num_sensed_uavs(0) {}

    SmartLidar(double resolution,
               CPolicy composition_policy,
               const std::vector<bool>& disappear_bools,
               const std::vector<std::string>& poi_subtypes,
               const std::vector<std::vector<std::string>>& agent_observable_subtypes,
               const std::vector<std::string>& accum_type,
               const std::vector<std::string>& measurement_type,
               const std::vector<double>& observation_radii,
               const std::vector<double>& default_values)
        : m_resolution(resolution),
          m_composition(composition_policy),
          m_disappear_bools(disappear_bools),
          m_poi_subtypes(poi_subtypes),
          m_agent_observable_subtypes(agent_observable_subtypes),
          m_accum_type(accum_type),
          m_measurement_type(measurement_type),
          m_observation_radii(observation_radii),
          m_default_values(default_values),
          m_num_sensed_uavs(0) {}

    [[nodiscard]] inline double measure(double distance, int agent_id) const {
        const std::string& mtype = m_measurement_type[agent_id];

        if (mtype == "inverse_distance_squared") {
            return 1.0 / std::max(0.001, distance * distance);
        } else if (mtype == "exponential_negative_distance") {
            return std::exp(-distance);
        } else if (mtype == "inverse_distance") {
            return 1.0 / std::max(0.001, distance);
        } else if (mtype == "distance_over_observation_radius") {
            return distance / m_observation_radii[agent_id];
        } else if (mtype == "one_minus_inverse_distance_over_observation_radius") {
            return 1.0 - distance / m_observation_radii[agent_id];
        }

        throw std::runtime_error("Measurement type for agent " + std::to_string(agent_id) + " is not defined!");
    }

    [[nodiscard]] std::vector<double> scan(const Agents& agents, const POIs& pois, int agent_idx) const {
        const std::size_t num_sectors = 360 / m_resolution;
        std::vector<std::vector<double>> poi_values(num_sectors);
        std::vector<std::vector<double>> rover_values(num_sectors);
        std::vector<std::vector<double>> uav_values(num_sectors);

        Agent agent = agents[agent_idx];
        AgentType my_type = agent->type();
        m_num_sensed_uavs = 0;

        // Observe POIs
        for (std::size_t poi_ind = 0; poi_ind < pois.size(); ++poi_ind) {
            auto& sensed_poi = pois[poi_ind];

            auto [angle, distance] = thyme::math::l2a(agent->position(), sensed_poi->position());
            // Match Python: if angle < 0, add 360
            if (angle < 0.0) angle += 360.0;

            if (distance > agent->obs_radius()) continue;

            // Check if agent can observe this POI type
            if (pois[poi_ind]->scope() == VisibilityScope::UAV_ONLY) {
                if (my_type == AgentType::Rover) {
                    // Rovers cannot observe hidden POIs, but can capture them
                    if (distance <= 1.0 && m_disappear_bools[poi_ind]) {
                        sensed_poi->set_captured(true);
                    }
                    continue;
                } else if (my_type == AgentType::UAV) {
                    // UAVs can observe specific subtypes
                    const std::string& poi_subtype = m_poi_subtypes[poi_ind];
                    if (!poi_subtype.empty()) {
                        const auto& observable = m_agent_observable_subtypes[agent_idx];
                        if (std::find(observable.begin(), observable.end(), poi_subtype) == observable.end()) {
                            continue;
                        }
                    }
                }
            }

            // Determine sector (matching Python logic)
            int sector;
            if (angle < 360.0) {
                sector = static_cast<int>(angle / m_resolution);
            } else {
                sector = 0;
            }

            // Add POI observation if not already captured
            if (!sensed_poi->captured()) {
                poi_values[sector].push_back(sensed_poi->value() * measure(distance, agent_idx));
            }
        }

        // Observe Agents
        for (std::size_t i = 0; i < agents.size(); ++i) {
            if (i == agent_idx) continue;

            auto& sensed_agent = agents[i];
            auto [angle, distance] = thyme::math::l2a(agent->position(), sensed_agent->position());
            // Match Python: if angle < 0, add 360
            if (angle < 0.0) angle += 360.0;

            if (distance > agent->obs_radius()) continue;

            // Determine sector (matching Python logic)
            int sector;
            if (angle < 360.0) {
                sector = static_cast<int>(angle / m_resolution);
            } else {
                sector = 0;
            }

            if (sensed_agent->type() == AgentType::Rover) {
                rover_values[sector].push_back(measure(distance, agent_idx));
            } else if (sensed_agent->type() == AgentType::UAV) {
                uav_values[sector].push_back(measure(distance, agent_idx));
                m_num_sensed_uavs++;
            }
        }

        // Encode state
        const double default_val = m_default_values[agent_idx];
        std::vector<double> state(num_sectors * 3, default_val);

        for (std::size_t i = 0; i < num_sectors; ++i) {
            const std::size_t num_rovers = rover_values[i].size();
            const std::size_t num_uavs = uav_values[i].size();
            const std::size_t num_pois = poi_values[i].size();

            const std::string& accum = m_accum_type[agent_idx];

            if (num_rovers > 0) {
                if (accum == "average") {
                    state[i] = m_composition->compose(rover_values[i], 0.0, num_rovers);
                } else if (accum == "sum") {
                    state[i] = m_composition->compose(rover_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(agent_idx));
                }
            }

            if (num_uavs > 0) {
                if (accum == "average") {
                    state[num_sectors + i] = m_composition->compose(uav_values[i], 0.0, num_uavs);
                } else if (accum == "sum") {
                    state[num_sectors + i] = m_composition->compose(uav_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(agent_idx));
                }
            }

            if (num_pois > 0) {
                if (accum == "average") {
                    state[num_sectors * 2 + i] = m_composition->compose(poi_values[i], 0.0, num_pois);
                } else if (accum == "sum") {
                    state[num_sectors * 2 + i] = m_composition->compose(poi_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(agent_idx));
                }
            }
        }

        return state;
    }

    [[nodiscard]] inline int num_sensed_uavs() const {
        return m_num_sensed_uavs;
    }

   private:
    double m_resolution;
    CPolicy m_composition;
    std::vector<bool> m_disappear_bools;
    std::vector<std::string> m_poi_subtypes;
    std::vector<std::vector<std::string>> m_agent_observable_subtypes;
    std::vector<std::string> m_accum_type;
    std::vector<std::string> m_measurement_type;
    std::vector<double> m_observation_radii;
    std::vector<double> m_default_values;
    mutable int m_num_sensed_uavs;
};

}  // namespace rover_domain

#endif
