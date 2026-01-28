#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_ICOMPOSITION
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_ICOMPOSITION

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
