#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES

#include <rover_domain/utilities/shared_wrapper.hpp>
#include <vector>
#include <string>

/*
 *
 * Forward declarations for agent types
 * And for enums
 *
 */
namespace rover_domain {
class IAgent;
using Agent = thyme::utilities::SharedWrap<IAgent>;
using Agents = std::vector<Agent>;
enum class AgentType{
    Rover,
    UAV
};

}  // namespace rover_domain

#endif
