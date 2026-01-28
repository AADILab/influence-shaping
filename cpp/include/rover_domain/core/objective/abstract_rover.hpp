#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_ABSTRACT_ROVER_CONSTRAINT
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_ABSTRACT_ROVER_CONSTRAINT

#include <rover_domain/core/interface/iobjective.hpp>

namespace rover_domain {

/*
 *
 * AbstractRoverObjective - base class for rover-specific objectives
 *
 */
class AbstractRoverObjective : public IObjective {
   public:
    // Default constructor
    AbstractRoverObjective()
        : m_coupling(1), m_is_rover_list() {}

    AbstractRoverObjective(int coupling, const std::vector<bool>& is_rover_list)
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

    [[nodiscard]] double step_score(const POIs& pois, const Agents& agents, int poi_idx, std::size_t t) const {
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

            // Calculate objective value as product of 1/dist for first m_coupling distances
            double score = static_cast<double>(m_coupling);
            for (int i = 0; i < m_coupling && i < dists.size(); ++i) {
                score *= 1.0 / dists[i];
            }

            return score;
        }

        return 0.0;
    }

   protected:
    int m_coupling;
    std::vector<bool> m_is_rover_list;
};

}  // namespace rover_domain

#endif