#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_IROVER
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_IROVER

#include <rover_domain/core/declare/agent_types.hpp>
#include <rover_domain/core/declare/entity_types.hpp>
#include <rover_domain/utilities/math/cartesian.hpp>
#include <vector>

namespace rover_domain {

class AutomaticParameters {
    public:
    AutomaticParameters() = default;
    AutomaticParameters(std::string timescale, std::string credit) {
        m_timescale = timescale;
        m_credit = credit;
    }
    std::string m_timescale;
    std::string m_credit;
};

class IndirectDifferenceParameters {
    public:
    IndirectDifferenceParameters(std::string type_, std::string assignment, std::vector<int> manual, AutomaticParameters automatic_parameters, bool add_G) {
        m_type = type_;
        m_assignment = assignment;
        m_manual = manual;
        m_automatic_parameters = automatic_parameters;
        m_add_G = add_G;
    }

    std::string m_type;
    std::string m_assignment;
    std::vector<int> m_manual;
    AutomaticParameters m_automatic_parameters;
    bool m_add_G;
};

class Bounds {
    public:
    Bounds(double low_x, double high_x, double low_y, double high_y) {
        m_low_x = low_x;
        m_high_x = high_x;
        m_low_y = low_y;
        m_high_y = high_y;
    }

    double m_low_x;
    double m_high_x;
    double m_low_y;
    double m_high_y;
};

/*
 *
 * agent interface
 *
 */
class IAgent {
    using Point = thyme::math::Point;
    using ActionType = std::vector<double>;
    using StateType = std::vector<double>;

   public:
    IAgent(
        Bounds bounds,
        IndirectDifferenceParameters indirect_difference_parameters,
        std::string reward_type,
        AgentType agent_type,
        double obs_radius = 1.0
    ) : m_bounds(bounds),
        m_indirect_difference_parameters(indirect_difference_parameters),
        m_reward_type(reward_type),
        m_type(agent_type),
        m_obs_radius(obs_radius) {};

    IAgent(IAgent&&) = default;
    IAgent(const IAgent&) = default;
    virtual ~IAgent() = default;

    void reset() {
        // std::cout << "IAgent::reset()" << std::endl;
        m_path.clear();
        }

    const Point& position() const { return m_position; }
    void set_position(double x, double y) {
        m_position.x = x;
        m_position.y = y;
        m_path.push_back(Point(x, y));
    }
    void update_position (double dx, double dy){
        set_position(m_position.x + dx, m_position.y + dy);
    }

    const double& obs_radius() const { return m_obs_radius; }

    const std::vector<Point>& path() const { return m_path; }

    void update() {
        // housekeeping.
        // delegate to tick for implementers.
        tick();
    }

    [[nodiscard]] virtual StateType scan(const Agents& agents, const POIs& pois, int agent_idx) const = 0;

    AgentType type() {
        // Give me the nominal type of this rover
        return m_type;
    }
    std::string reward_type() {
        return m_reward_type;
    }

    IndirectDifferenceParameters indirect_difference_parameters() {
        return m_indirect_difference_parameters;
    }

    Bounds bounds() {
        return m_bounds;
    }

    // [TODO] temp cppyy super().__init__() fix
    virtual void act(const ActionType&) {}

   protected:
    virtual void tick() {}

   private:
    std::string m_reward_type;
    AgentType m_type;
    double m_obs_radius;
    Point m_position;
    std::vector<Point> m_path;
    IndirectDifferenceParameters m_indirect_difference_parameters;
    Bounds m_bounds;
};

} // namespace rover_domain

#endif
