#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES

#include <roverdomain/utilities/shared_wrapper.hpp>

/*
*
* Agent abstraction
*
*/
namespace rovers {
class IRover;
using Agent = thyme::utilities::SharedWrap<IRover>;
}  // namespace rovers

#endif
