#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_PACK
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_PACK

#include <functional>
#include <rover_domain/core/detail/agent_types.hpp>
#include <rover_domain/core/detail/entity_types.hpp>
#include <rover_domain/core/rover/irover.hpp>
#include <rover_domain/core/poi/ipoi.hpp>
#include <rover_domain/utilities/ranges.hpp>

/*
 *
 * Parameter packs for common aggregations
 *
 */
namespace rovers {

struct AgentPack {
    AgentPack(int agent_index, const std::vector<Agent>& agents,
              const std::vector<Entity>& entities)
        : agent_index(agent_index), agents(agents), entities(entities) {}
    int agent_index;
    std::vector<Agent> agents;
    std::vector<Entity> entities;
};

// AgentPack from_filter(const AgentPack& pack, std::function<bool(const Agent&)> predicate) {
//     return {pack.agent, thyme::utilities::filter(pack.agents, predicate), pack.entities};
// }

struct EntityPack {
    EntityPack(const Entity& entity, const std::vector<Agent>& agents,
               const std::vector<Entity>& entities)
        : entity(entity), agents(agents), entities(entities) {}
    Entity entity;
    std::vector<Agent> agents;
    std::vector<Entity> entities;
};

EntityPack from_filter(const EntityPack& pack, std::function<bool(const Entity&)> predicate) {
    return {pack.entity, pack.agents, thyme::utilities::filter(pack.entities, predicate)};
}
}  // namespace rovers

#endif
