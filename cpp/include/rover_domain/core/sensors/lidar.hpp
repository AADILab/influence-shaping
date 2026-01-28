#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_LIDAR
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_LIDAR

#include <Eigen/Dense>
#include <numeric>
#include <rover_domain/core/poi/default_poi.hpp>
#include <rover_domain/core/rover/rover.hpp>
#include <rover_domain/utilities/math/norms.hpp>
#include <rover_domain/core/interface/isensor.hpp>
#include <rover_domain/core/composition/density.hpp>
#include <vector>

namespace rover_domain {

/*
 *
 * Lidar
 * Senses agents and POIs
 *
 */
template <typename CompositionPolicy = Density>
class Lidar : public ISensor {
    using CPolicy = thyme::utilities::SharedWrap<CompositionPolicy>;

   public:
    Lidar(double resolution = 90, CPolicy composition_policy = CompositionPolicy())
        : m_resolution(resolution), m_composition(composition_policy) {}

    [[nodiscard]] Eigen::MatrixXd scan(const Agents& agents, const POIs& pois, int agent_idx) const {
        // std::cout << "Lidar::scan()" << std::endl;
        const std::size_t num_sectors = 360 / m_resolution;
        std::vector<std::vector<double>> poi_values(num_sectors), rover_values(num_sectors);
        auto& rover = agents[agent_idx];  // convenient handle

        // observe pois
        for (const auto& sensed_poi : pois) {
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
        for (int i = 0; i < agents.size(); ++i) {
            // Do not observe yourself
            if (i == agent_idx) continue;

            auto& sensed_rover = agents[i];  //convenient handle
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

}  // namespace rover_domain

#endif