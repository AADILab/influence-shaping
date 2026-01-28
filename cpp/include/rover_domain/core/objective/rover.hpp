#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER_CONSTRAINT
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER_CONSTRAINT

#include <rover_domain/core/objective/abstract_rover.hpp>

namespace rover_domain {

/*
 *
 * RoverObjective - objective based on final positions
 *
 */
class RoverObjective : public AbstractRoverObjective {
   public:
    // Default constructor
    RoverObjective() : AbstractRoverObjective() {}

    using AbstractRoverObjective::AbstractRoverObjective;

    [[nodiscard]] double score(const POIs& pois, const Agents& agents, int poi_idx) const override {
        // No agents means constraint is not satisfied
        if (agents.size() == 0) {
            return 0.0;
        }

        // Get final timestep
        std::size_t t_final = agents[0]->path().size() - 1;
        return step_score(pois, agents, poi_idx, t_final);
    }
};

}  // namespace rover_domain

#endif
