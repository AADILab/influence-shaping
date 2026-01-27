#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES

#include <rover_domain/utilities/shared_wrapper.hpp>

/*
 *
 * Forward declarations for agent types
 *
 */
namespace rover_domain {
class IAgent;
using Agent = thyme::utilities::SharedWrap<IAgent>;
using Agents = std::vector<Agent>;
}  // namespace rover_domain

#endif
