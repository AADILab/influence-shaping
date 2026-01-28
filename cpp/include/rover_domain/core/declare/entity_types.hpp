#ifndef BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI_TYPES
#define BASIL_ENVIRONMENTS_ROVER_DOMAIN_POI_TYPES

#include <rover_domain/utilities/shared_wrapper.hpp>

/*
 *
 * Forward declarations for POI types
 *
 */
namespace rover_domain {
class IPOI;
using POI = thyme::utilities::SharedWrap<IPOI>;
using POIs = std::vector<POI>;
}  // namespace rover_domain

#endif
