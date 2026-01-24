#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_PACK
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_PACK

#include <functional>
#include <rover_domain/core/detail/agent_types.hpp>
#include <rover_domain/core/detail/entity_types.hpp>
#include <rover_domain/core/rover/irover.hpp>
#include <rover_domain/core/poi/ipoi.hpp>

/*
 *
 * Parameter packs for common aggregations
 *
 */
namespace rover_domain {

struct AgentPack {
    AgentPack(int agent_index, const std::vector<Agent>& agents,
              const std::vector<POI>& entities)
        : agent_index(agent_index), agents(agents), entities(entities) {}
    int agent_index;
    std::vector<Agent> agents;
    std::vector<POI> entities;
};

struct POIPack {
    POIPack(const POI& entity, const std::vector<Agent>& agents,
               const std::vector<POI>& entities)
        : entity(entity), agents(agents), entities(entities) {}
    POI entity;
    std::vector<Agent> agents;
    std::vector<POI> entities;
};

}  // namespace rover_domain

#endif
