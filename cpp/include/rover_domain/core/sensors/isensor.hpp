#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ISENSOR
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ISENSOR

#include <Eigen/Dense>
#include <rover_domain/core/detail/pack.hpp>

namespace rover_domain {

/*
 *
 * sensor interface for bindings
 *
 */
class ISensor {
   public:
    [[nodiscard]] virtual Eigen::MatrixXd scan(const AgentPack& pack) const = 0;
    virtual ~ISensor() = default;
};

}  // namespace rover_domain
#endif