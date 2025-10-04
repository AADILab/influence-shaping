#ifndef THYME_ENVIRONMENTS_ROVERS_ROVER
#define THYME_ENVIRONMENTS_ROVERS_ROVER

#include <Eigen/Dense>
#include <iostream>
#include <rovers/core/detail/agent_types.hpp>
#include <rovers/core/detail/entity_types.hpp>
#include <rovers/core/detail/pack.hpp>
#include <rovers/core/rewards/global.hpp>
#include <rovers/utilities/math/cartesian.hpp>
#include <vector>

namespace rovers {

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

/*
 *
 * Default boilerplate rover
 *
 */
template <typename SensorType, typename ActionSpace, typename RewardType = rewards::Global>
class Rover final : public IRover {
    using SType = thyme::utilities::SharedWrap<SensorType>;
    using RType = thyme::utilities::SharedWrap<RewardType>;
    using ActionType = Eigen::MatrixXd;
   public:
    Rover(Bounds bounds, IndirectDifferenceParameters indirect_difference_parameters, std::string reward_type, std::string type_, double obs_radius = 1.0, SType sensor = SensorType(), RType reward = RewardType())
        : IRover(bounds, indirect_difference_parameters, reward_type, type_, obs_radius), m_sensor(sensor), m_reward(reward) {}
    // NOTE: This is commented out because I couldn't get it to work properly, but left as dead code to help me later if I need to get it working
    // Rover(const Rover& rover)
    //     : IRover(rover.indirect_difference_parameters(), rover.reward_type(), rover.type(), rover.obs_radius()), m_sensor(SensorType()), m_reward(RewardType()) {}
    [[nodiscard]] virtual Eigen::MatrixXd scan(const AgentPack& pack) const override {
        // std::cout << "Rover::scan()" << std::endl;
        return m_sensor->scan(pack);
    }
    [[nodiscard]] virtual double reward(const AgentPack& pack) const override {
        // each aget gets a reward set here but only nominally so the reward computer knows
        // what to do
        // but each agent is not comjputing its own reward
        // std::cout << "Rover::reward()" << std::endl;
        return m_reward->compute(pack);
    }
    void act(const ActionType& action) override {
        // default, move in x and y
        assert(action.rows() >= 2);
        auto act = static_cast<Eigen::Vector2d>(action);
        update_position(act[0], act[1]);
    }

   private:
    SType m_sensor;
    RType m_reward;
};

/*
 *
 * Example of bringing in a new Rover from the python bindings
 *
 */
// class Drone final : public IRover {
//    public:
//     Drone(double obs_radius = 1.0) : IRover(obs_radius) {}

//     [[nodiscard]] virtual Eigen::MatrixXd scan(const AgentPack&) const override { return {}; }
//     [[nodiscard]] virtual double reward(const AgentPack&) const override { return 0; }
//     void act(const Eigen::MatrixXd&) override { }
// };

}  // namespace rovers

#endif
