#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_POI_TYPES
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_POI_TYPES

#include <rover_domain/utilities/shared_wrapper.hpp>

/*
 *
 * Entity abstraction
 *
 */
namespace rovers {
class IPOI;
using Entity = thyme::utilities::SharedWrap<IPOI>;
}  // namespace rovers

#endif
