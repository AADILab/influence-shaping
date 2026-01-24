#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_LIDAR
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_LIDAR

#include <Eigen/Dense>
#include <numeric>
#include <roverdomain/core/detail/pack.hpp>
#include <roverdomain/core/poi/poi.hpp>
#include <roverdomain/core/rover/rover.hpp>
#include <roverdomain/utilities/math/norms.hpp>
#include <roverdomain/core/sensors/isensor.hpp>
// #include <ranges> // changed for python branch
#include <vector>

namespace rovers {

/*
 *
 * Lidar composition strategies
 *
 */
class Density {
   public:
    // template <std::ranges::range Range, typename Tp, typename Up>
    template <typename Range, typename Tp, typename Up>
    inline Tp compose(const Range& range, Tp init, Up scale) const {
        return std::accumulate(std::begin(range), std::end(range), init) / scale;
    }
};

class Closest {
   public:
    // template <std::ranges::range Range, typename Tp, typename Up>
    template <typename Range, typename Tp, typename Up>
    inline Tp compose(const Range& range, Tp, Up) const {
        return *std::max_element(std::begin(range), std::end(range));
    }
};

/*
 *
 *  composition strategy interface for bindings
 *
 */
class ISensorComposition {
   public:
    virtual inline double compose(const std::vector<double>, double, double) = 0;
    virtual ~ISensorComposition() = default;
};

/*
 *
 * Lidar
 *
 */
template <typename CompositionPolicy = Density>
class Lidar : public ISensor {
    using CPolicy = thyme::utilities::SharedWrap<CompositionPolicy>;

   public:
    Lidar(double resolution = 90, CPolicy composition_policy = CompositionPolicy())
        : m_resolution(resolution), m_composition(composition_policy) {}

    [[nodiscard]] Eigen::MatrixXd scan(const AgentPack& pack) const {
        // std::cout << "Lidar::scan()" << std::endl;
        const std::size_t num_sectors = 360 / m_resolution;
        std::vector<std::vector<double>> poi_values(num_sectors), rover_values(num_sectors);
        auto& rover = pack.agents[pack.agent_index];  // convenient handle

        // observe pois
        for (const auto& sensed_poi : pack.entities) {
            if (sensed_poi->observed()) continue;
            auto [angle, distance] = thyme::math::l2a(rover->position(), sensed_poi->position());
            if (distance > rover->obs_radius()) continue;

            int sector;
            if (angle < 360.0) sector = angle / m_resolution;
            else sector = 0;
            poi_values[sector].push_back(sensed_poi->value() /
                                         std::max(0.001, distance * distance));
        }

        // observe rovers
        for (int i = 0; i < pack.agents.size(); ++i) {
            // Do not observe yourself
            if (i == pack.agent_index) continue;

            auto& sensed_rover = pack.agents[i];  //convenient handle
            auto [angle, distance] = thyme::math::l2a(rover->position(), sensed_rover->position());
            if (distance > rover->obs_radius()) continue;

            int sector;
            if (angle < 360.0) sector = angle / m_resolution;
            else sector = 0;
            rover_values[sector].push_back(1.0 / std::max(0.001, distance * distance));
        }

        // encode state
        Eigen::MatrixXd state(num_sectors * 2, 1);

        // For each sector
        for (std::size_t i = 0; i < num_sectors; ++i) {
            const std::size_t& num_rovers = rover_values[i].size();
            const std::size_t& num_poi = poi_values[i].size();
            state(i) = state(num_sectors + i) = -1.0;

            if (num_rovers > 0)
                state(i) = m_composition->compose(rover_values[i], 0.0, num_rovers);
            if (num_poi > 0)
                state(num_sectors + i) = m_composition->compose(poi_values[i], 0.0, num_poi);
        }
        return state;
    }

   private:
    double m_resolution;
    CPolicy m_composition;
};

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
               const std::vector<std::string>& agent_types,
               const std::vector<std::string>& poi_types,
               const std::vector<bool>& disappear_bools,
               const std::vector<std::string>& poi_subtypes,
               const std::vector<std::vector<std::string>>& agent_observable_subtypes,
               const std::vector<std::string>& accum_type,
               const std::vector<std::string>& measurement_type,
               const std::vector<double>& observation_radii,
               const std::vector<double>& default_values)
        : m_resolution(resolution),
          m_composition(composition_policy),
          m_agent_types(agent_types),
          m_poi_types(poi_types),
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

    [[nodiscard]] Eigen::MatrixXd scan(const AgentPack& pack) const {
        const std::size_t num_sectors = 360 / m_resolution;
        std::vector<std::vector<double>> poi_values(num_sectors);
        std::vector<std::vector<double>> rover_values(num_sectors);
        std::vector<std::vector<double>> uav_values(num_sectors);

        auto& agent = pack.agents[pack.agent_index];
        const std::string& my_type = m_agent_types[pack.agent_index];
        m_num_sensed_uavs = 0;

        // Observe POIs
        for (std::size_t poi_ind = 0; poi_ind < pack.entities.size(); ++poi_ind) {
            auto& sensed_poi = pack.entities[poi_ind];

            auto [angle, distance] = thyme::math::l2a(agent->position(), sensed_poi->position());
            // Match Python: if angle < 0, add 360
            if (angle < 0.0) angle += 360.0;

            if (distance > agent->obs_radius()) continue;

            // Check if agent can observe this POI type
            if (m_poi_types[poi_ind] == "hidden") {
                if (my_type == "rover") {
                    // Rovers cannot observe hidden POIs, but can capture them
                    if (distance <= 1.0 && m_disappear_bools[poi_ind]) {
                        sensed_poi->set_observed(true);
                    }
                    continue;
                } else if (my_type == "uav") {
                    // UAVs can observe specific subtypes
                    const std::string& poi_subtype = m_poi_subtypes[poi_ind];
                    if (!poi_subtype.empty()) {
                        const auto& observable = m_agent_observable_subtypes[pack.agent_index];
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
            if (!sensed_poi->observed()) {
                poi_values[sector].push_back(sensed_poi->value() * measure(distance, pack.agent_index));
            }
        }

        // Observe Agents
        for (std::size_t i = 0; i < pack.agents.size(); ++i) {
            if (i == pack.agent_index) continue;

            auto& sensed_agent = pack.agents[i];
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

            if (m_agent_types[i] == "rover") {
                rover_values[sector].push_back(measure(distance, pack.agent_index));
            } else if (m_agent_types[i] == "uav") {
                uav_values[sector].push_back(measure(distance, pack.agent_index));
                m_num_sensed_uavs++;
            }
        }

        // Encode state
        Eigen::MatrixXd state(num_sectors * 3, 1);
        const double default_val = m_default_values[pack.agent_index];
        state.setConstant(default_val);

        for (std::size_t i = 0; i < num_sectors; ++i) {
            const std::size_t num_rovers = rover_values[i].size();
            const std::size_t num_uavs = uav_values[i].size();
            const std::size_t num_pois = poi_values[i].size();

            const std::string& accum = m_accum_type[pack.agent_index];

            if (num_rovers > 0) {
                if (accum == "average") {
                    state(i) = m_composition->compose(rover_values[i], 0.0, num_rovers);
                } else if (accum == "sum") {
                    state(i) = m_composition->compose(rover_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(pack.agent_index));
                }
            }

            if (num_uavs > 0) {
                if (accum == "average") {
                    state(num_sectors + i) = m_composition->compose(uav_values[i], 0.0, num_uavs);
                } else if (accum == "sum") {
                    state(num_sectors + i) = m_composition->compose(uav_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(pack.agent_index));
                }
            }

            if (num_pois > 0) {
                if (accum == "average") {
                    state(num_sectors * 2 + i) = m_composition->compose(poi_values[i], 0.0, num_pois);
                } else if (accum == "sum") {
                    state(num_sectors * 2 + i) = m_composition->compose(poi_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(pack.agent_index));
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
    std::vector<std::string> m_agent_types;
    std::vector<std::string> m_poi_types;
    std::vector<bool> m_disappear_bools;
    std::vector<std::string> m_poi_subtypes;
    std::vector<std::vector<std::string>> m_agent_observable_subtypes;
    std::vector<std::string> m_accum_type;
    std::vector<std::string> m_measurement_type;
    std::vector<double> m_observation_radii;
    std::vector<double> m_default_values;
    mutable int m_num_sensed_uavs;
};

/*
 *
 * RoverLidar - differentiates between rovers and UAVs only (no POI sensing)
 *
 */
template <typename CompositionPolicy = Density>
class RoverLidar : public ISensor {
    using CPolicy = thyme::utilities::SharedWrap<CompositionPolicy>;

   public:
    // Default constructor
    RoverLidar()
        : m_resolution(90),
          m_composition(CompositionPolicy()),
          m_num_sensed_uavs(0) {}

    RoverLidar(double resolution,
               CPolicy composition_policy,
               const std::vector<std::string>& agent_types,
               const std::vector<std::string>& accum_type,
               const std::vector<std::string>& measurement_type,
               const std::vector<double>& observation_radii,
               const std::vector<double>& default_values)
        : m_resolution(resolution),
          m_composition(composition_policy),
          m_agent_types(agent_types),
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

    [[nodiscard]] Eigen::MatrixXd scan(const AgentPack& pack) const {
        const std::size_t num_sectors = 360 / m_resolution;
        std::vector<std::vector<double>> rover_values(num_sectors);
        std::vector<std::vector<double>> uav_values(num_sectors);

        auto& agent = pack.agents[pack.agent_index];
        m_num_sensed_uavs = 0;

        // Observe Agents only (no POI sensing)
        for (std::size_t i = 0; i < pack.agents.size(); ++i) {
            if (i == pack.agent_index) continue;

            auto& sensed_agent = pack.agents[i];
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

            if (m_agent_types[i] == "rover") {
                rover_values[sector].push_back(measure(distance, pack.agent_index));
            } else if (m_agent_types[i] == "uav") {
                uav_values[sector].push_back(measure(distance, pack.agent_index));
                m_num_sensed_uavs++;
            }
        }

        // Encode state (only rovers and UAVs, no POIs)
        Eigen::MatrixXd state(num_sectors * 2, 1);
        const double default_val = m_default_values[pack.agent_index];
        state.setConstant(default_val);

        for (std::size_t i = 0; i < num_sectors; ++i) {
            const std::size_t num_rovers = rover_values[i].size();
            const std::size_t num_uavs = uav_values[i].size();

            const std::string& accum = m_accum_type[pack.agent_index];

            if (num_rovers > 0) {
                if (accum == "average") {
                    state(i) = m_composition->compose(rover_values[i], 0.0, num_rovers);
                } else if (accum == "sum") {
                    state(i) = m_composition->compose(rover_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(pack.agent_index));
                }
            }

            if (num_uavs > 0) {
                if (accum == "average") {
                    state(num_sectors + i) = m_composition->compose(uav_values[i], 0.0, num_uavs);
                } else if (accum == "sum") {
                    state(num_sectors + i) = m_composition->compose(uav_values[i], 0.0, 1.0);
                } else {
                    throw std::runtime_error("Invalid accumulation type '" + accum + "' for agent " + std::to_string(pack.agent_index));
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
    std::vector<std::string> m_agent_types;
    std::vector<std::string> m_accum_type;
    std::vector<std::string> m_measurement_type;
    std::vector<double> m_observation_radii;
    std::vector<double> m_default_values;
    mutable int m_num_sensed_uavs;
};

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

    UavDistanceLidar(const std::vector<std::string>& agent_types)
        : m_agent_types(agent_types),
          m_num_sensed_uavs(0) {}

    [[nodiscard]] Eigen::MatrixXd scan(const AgentPack& pack) const {
        auto& agent = pack.agents[pack.agent_index];
        m_num_sensed_uavs = 0;
        std::vector<double> distances;

        // Iterate through all agents
        for (std::size_t i = 0; i < pack.agents.size(); ++i) {
            // Only process UAVs
            if (m_agent_types[i] == "uav") {
                auto& sensed_agent = pack.agents[i];
                double distance = thyme::math::l2_norm(agent->position(), sensed_agent->position());

                if (distance <= agent->obs_radius()) {
                    distances.push_back(distance / agent->obs_radius());
                    m_num_sensed_uavs++;
                } else {
                    distances.push_back(-1.0);
                }
            }
        }

        // Convert to Eigen column vector
        Eigen::MatrixXd state(distances.size(), 1);
        for (std::size_t i = 0; i < distances.size(); ++i) {
            state(i) = distances[i];
        }

        return state;
    }

    [[nodiscard]] inline int num_sensed_uavs() const {
        return m_num_sensed_uavs;
    }

   private:
    std::vector<std::string> m_agent_types;
    mutable int m_num_sensed_uavs;
};

}  // namespace rovers

#endif