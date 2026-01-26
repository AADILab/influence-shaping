#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_POI_TYPES
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_POI_TYPES

#include <rover_domain/utilities/shared_wrapper.hpp>

/*
 *
 * POI abstraction
 *
 */
namespace rover_domain {
class IPOI;
using POI = thyme::utilities::SharedWrap<IPOI>;
using POIs = std::vector<POI>;
}  // namespace rover_domain

#endif
