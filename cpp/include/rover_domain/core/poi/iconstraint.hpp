#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_POI_ICONSTRAINT
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_POI_ICONSTRAINT

#include <rover_domain/core/rover/rover.hpp>
#include <rover_domain/core/detail/agent_types.hpp>
#include <rover_domain/core/detail/entity_types.hpp>
#include <rover_domain/utilities/math/norms.hpp>
#include <algorithm>
#include <vector>
#include <limits>

namespace rover_domain {

/*
 *
 * Constraint interface for bindings
 *
 */
class IConstraint {
   public:
    [[nodiscard]] virtual double is_satisfied(const POIs& pois, const Agents& agents, int poi_idx) const = 0;
    virtual ~IConstraint() = default;
};

/*
 *
 * AbstractRoverConstraint - base class for rover-specific constraints
 *
 */
class AbstractRoverConstraint : public IConstraint {
   public:
    // Default constructor
    AbstractRoverConstraint()
        : m_coupling(1), m_is_rover_list() {}

    AbstractRoverConstraint(int coupling, const std::vector<bool>& is_rover_list)
        : m_coupling(coupling), m_is_rover_list(is_rover_list) {}

    [[nodiscard]] bool captured(double dist, const Agent& agent, const POI& entity) const {
        // Check if captured by capture radius or observation radii
        if (entity->capture_radius() != -1.0 && dist <= entity->capture_radius()) {
            return true;
        } else if (dist <= agent->obs_radius() && dist <= entity->obs_radius()) {
            return true;
        }
        return false;
    }

    [[nodiscard]] double step_is_satisfied(const POIs& pois, const Agents& agents, int poi_idx, std::size_t t) const {
        int count = 0;
        std::vector<double> dists;
        bool constraint_satisfied = false;

        for (const auto& agent : agents) {
            if (agent->type() == "rover") {
                double dist;
                // Check for counterfactual removal (position [-1, -1])
                if (agent->path()[t].x == -1 && agent->path()[t].y == -1) {
                    dist = std::numeric_limits<double>::infinity();
                } else {
                    dist = thyme::math::l2_norm(agent->path()[t], pois[poi_idx]->position());
                }

                dists.push_back(dist);

                if (captured(dist, agent, pois[poi_idx])) {
                    count++;
                    if (count >= m_coupling) {
                        constraint_satisfied = true;
                    }
                }
            }
        }

        if (constraint_satisfied) {
            // Sort distances
            std::sort(dists.begin(), dists.end());

            // Apply max(1.0, dist) to each distance
            for (auto& dist : dists) {
                dist = std::max(1.0, dist);
            }

            // Calculate constraint value as product of 1/dist for first m_coupling distances
            double constraint_value = static_cast<double>(m_coupling);
            for (int i = 0; i < m_coupling && i < dists.size(); ++i) {
                constraint_value *= 1.0 / dists[i];
            }

            return constraint_value;
        }

        return 0.0;
    }

   protected:
    int m_coupling;
    std::vector<bool> m_is_rover_list;
};

/*
 *
 * RoverConstraint - constraint based on final positions
 *
 */
class RoverConstraint : public AbstractRoverConstraint {
   public:
    // Default constructor
    RoverConstraint() : AbstractRoverConstraint() {}

    using AbstractRoverConstraint::AbstractRoverConstraint;

    [[nodiscard]] double is_satisfied(const POIs& pois, const Agents& agents, int poi_idx) const override {
        // No agents means constraint is not satisfied
        if (agents.size() == 0) {
            return 0.0;
        }

        // Get final timestep
        std::size_t t_final = agents[0]->path().size() - 1;
        return step_is_satisfied(pois, agents, poi_idx, t_final);
    }
};

/*
 *
 * RoverSequenceConstraint - constraint based on closest positions in paths
 *
 */
class RoverSequenceConstraint : public AbstractRoverConstraint {
   public:
    // Default constructor
    RoverSequenceConstraint() : AbstractRoverConstraint() {}

    using AbstractRoverConstraint::AbstractRoverConstraint;

    [[nodiscard]] double is_satisfied(const POIs& pois, const Agents& agents, int poi_idx) const override {
        // No agents means constraint is not satisfied
        if (agents.size() == 0) {
            return 0.0;
        }

        // Find maximum satisfaction across all timesteps
        std::vector<double> steps;
        std::size_t path_size = agents[0]->path().size();

        for (std::size_t t = 0; t < path_size; ++t) {
            steps.push_back(step_is_satisfied(pois, agents, poi_idx, t));
        }

        return *std::max_element(steps.begin(), steps.end());
    }
};

}  // namespace rover_domain

#endif