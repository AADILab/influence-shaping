#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_SEQUENCE_CONSTRAINT
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_SEQUENCE_CONSTRAINT

#include <rover_domain/core/constraint/abstract_rover.hpp>

namespace rover_domain {

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
