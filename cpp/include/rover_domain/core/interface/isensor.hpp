#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_ISENSOR
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_ISENSOR

#include <rover_domain/core/declare/agent_types.hpp>
#include <rover_domain/core/declare/entity_types.hpp>

namespace rover_domain {

/*
 *
 * sensor interface for bindings
 *
 */
class ISensor {
   public:
    [[nodiscard]] virtual std::vector<double> scan(const Agents& agents, const POIs& pois, int agent_idx) const = 0;
    virtual ~ISensor() = default;
};

}  // namespace rover_domain
#endif