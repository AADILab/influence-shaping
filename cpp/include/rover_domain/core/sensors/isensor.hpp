#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ISENSOR
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ISENSOR

#include <Eigen/Dense>
#include <rover_domain/core/detail/agent_types.hpp>
#include <rover_domain/core/detail/entity_types.hpp>

namespace rover_domain {

/*
 *
 * sensor interface for bindings
 *
 */
class ISensor {
   public:
    [[nodiscard]] virtual Eigen::MatrixXd scan(const Agents& agents, const POIs& pois, int agent_idx) const = 0;
    virtual ~ISensor() = default;
};

}  // namespace rover_domain
#endif