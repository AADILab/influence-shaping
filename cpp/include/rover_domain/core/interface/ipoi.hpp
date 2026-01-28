#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_IPOI
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_IPOI

#include <rover_domain/utilities/math/cartesian.hpp>
#include <rover_domain/core/declare/agent_types.hpp>
#include <rover_domain/core/declare/entity_types.hpp>

namespace rover_domain {

/*
 *
 * POI interface
 *
 */
class IPOI {
    using Point = thyme::math::Point;

   public:
    IPOI(double value, double obs_radius, double capture_radius) : m_value(value), m_obs_radius(obs_radius), m_capture_radius(capture_radius) {}
    virtual ~IPOI() = default;

    const Point& position() const { return m_position; }
    void set_position(double x, double y) {
        m_position.x = x;
        m_position.y = y;
    }

    const double& value() const { return m_value; }
    const double& obs_radius() const { return m_obs_radius; }
    const double& capture_radius() const { return m_capture_radius; }

    void set_observed(bool observed) {
        // std::cout << "IPOI::set_observed()" << std::endl;
        m_observed = observed;
    }
    const bool& observed() const { return m_observed; }

    void update() {
        // housekeeping.
        // delegate to tick for implementers.
        tick();
    }

    [[nodiscard]] virtual double score(const POIs&, const Agents&, int poi_idx) const = 0;

   protected:
    virtual void tick() {}

   private:
    Point m_position;
    double m_value;

    double m_obs_radius;
    double m_capture_radius;
    bool m_observed{false};
    // TODO: Add another class variable, like m_value_achieved or something like that
    // Basically the same function as m_observed, but it is a floating point value
    // That tells us how much of the reward from this poi has already been gained
    // And it starts at 0. Any time a rover gets closer (at a given timestep),
    // we'll raise the value up
    // And then we need to track some delta (so basically, how much has that value changed)
    // from timestep to timestep, so maybe easiest thing is something like
    // m_highest_value_achived
    // m_last_highest_value_achived
    // Maybe at each call for tick() we update these variables
    // Then we compute a stepwise reward based on the delta between the two variables
};

} // namespace rover_domain

#endif