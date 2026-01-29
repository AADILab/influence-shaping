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

// Helper functions for string conversion
inline std::string to_string(AgentType type) {
    switch(type) {
        case AgentType::Rover: return std::string("rover");
        case AgentType::UAV: return "uav";
        default: throw std::invalid_argument("Unknown enum for agent type: " + std::to_string(static_cast<int>(type)));
    }
}

inline AgentType agent_type_from_string(const std::string& str) {
    if (str == "rover") return AgentType::Rover;
    if (str == "uav") return AgentType::UAV;
    throw std::invalid_argument("Unknown agent type: " + str);
}

}  // namespace rover_domain

#endif
