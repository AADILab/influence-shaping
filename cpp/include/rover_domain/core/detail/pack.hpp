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
              const std::vector<Entity>& entities)
        : agent_index(agent_index), agents(agents), entities(entities) {}
    int agent_index;
    std::vector<Agent> agents;
    std::vector<Entity> entities;
};

struct EntityPack {
    EntityPack(const Entity& entity, const std::vector<Agent>& agents,
               const std::vector<Entity>& entities)
        : entity(entity), agents(agents), entities(entities) {}
    Entity entity;
    std::vector<Agent> agents;
    std::vector<Entity> entities;
};

}  // namespace rover_domain

#endif
