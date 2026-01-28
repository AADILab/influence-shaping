#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER_SEQUENCE_CONSTRAINT
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER_SEQUENCE_CONSTRAINT

#include <rover_domain/core/objective/abstract_rover.hpp>

namespace rover_domain {

/*
 *
 * RoverSequenceObjective - objective based on closest positions in paths
 *
 */
class RoverSequenceObjective : public AbstractRoverObjective {
   public:
    // Default constructor
    RoverSequenceObjective() : AbstractRoverObjective() {}

    using AbstractRoverObjective::AbstractRoverObjective;

    [[nodiscard]] double score(const POIs& pois, const Agents& agents, int poi_idx) const override {
        // No agents means constraint is not satisfied
        if (agents.size() == 0) {
            return 0.0;
        }

        // Find maximum score across all timesteps
        std::vector<double> steps;
        std::size_t path_size = agents[0]->path().size();

        for (std::size_t t = 0; t < path_size; ++t) {
            steps.push_back(step_score(pois, agents, poi_idx, t));
        }

        return *std::max_element(steps.begin(), steps.end());
    }
};

}  // namespace rover_domain

#endif
