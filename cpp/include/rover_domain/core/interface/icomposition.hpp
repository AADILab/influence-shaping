#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ICOMPOSITION
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ICOMPOSITION

#include <vector>

namespace rover_domain {

/*
 *
 *  composition strategy interface for bindings
 *
 */
class ISensorComposition {
   public:
    virtual inline double compose(const std::vector<double>, double, double) = 0;
    virtual ~ISensorComposition() = default;
};

}  // namespace rover_domain

#endif
