#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_IROVER
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_IROVER

#include <Eigen/Dense>
#include <iostream>
#include <rover_domain/core/detail/agent_types.hpp>
#include <rover_domain/core/detail/entity_types.hpp>
#include <rover_domain/utilities/math/cartesian.hpp>
#include <vector>

namespace rovers {

struct AgentPack;

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
 * rover interface
 *
 */
class IRover {
    using Point = thyme::math::Point;
    using ActionType = Eigen::MatrixXd;
    using StateType = Eigen::MatrixXd;

   public:
    IRover(Bounds bounds, IndirectDifferenceParameters indirect_difference_parameters, std::string reward_type, std::string type_, double obs_radius = 1.0) : m_bounds(bounds), m_indirect_difference_parameters(indirect_difference_parameters), m_reward_type(reward_type), m_type(type_), m_obs_radius(obs_radius) {};
    IRover(IRover&&) = default;
    IRover(const IRover&) = default;
    virtual ~IRover() = default;

    void reset() {
        // std::cout << "IRover::reset()" << std::endl;
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

    [[nodiscard]] virtual StateType scan(const AgentPack&) const = 0;
    [[nodiscard]] virtual double reward(const AgentPack&) const = 0;

    std::string type() {
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
    std::string m_type;
    double m_obs_radius;
    Point m_position;
    std::vector<Point> m_path;
    IndirectDifferenceParameters m_indirect_difference_parameters;
    Bounds m_bounds;
};

} // namespace rovers

#endif
