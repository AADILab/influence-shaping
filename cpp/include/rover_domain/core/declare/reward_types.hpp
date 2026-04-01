#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_REWARD_TYPES
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_REWARD_TYPES

#include <rover_domain/utilities/shared_wrapper.hpp>
#include <vector>

/*
 *
 * Forward declarations for reward types
 *
 */
namespace rover_domain {

// Global and difference rewards require no additional parameters
struct GlobalReward {};
struct DifferenceReward {};

// Indirect difference rewards require different parameters depending
// on how they are configured
// Static mode can be manual or automatic
struct IDStaticManual {
    std::vector<int> manual;
};
struct IDStaticAutomatic {
    enum class Credit { Local, WinnerTakesAll };
    Credit credit;
};
using IDStatic = std::variant<IDStaticManual, IDStaticAutomatic>;
// Dynamic mode is automatic by default. Just needs the credit assignment mechanism
struct IDDynamic {
    enum class Credit { Local, WinnerTakesAll, System, Difference };
    Credit credit;
};
// Adaptive mode
struct IDAdaptive {
    int N_agents;
    int n_timesteps;
};

// Indirect difference reward struct is then built depending on parameters needed
struct IndirectDifferenceReward {
    enum class Mode { Static, Dynamic, Adaptive };
    Mode mode;
    std::variant<IDStatic, IDDynamic, IDAdaptive> params;
    bool add_G = false;
};

using RewardSpec = std::variant<GlobalReward, DifferenceReward, IndirectDifferenceReward>;

}  // namespace rover_domain


#endif