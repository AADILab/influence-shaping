#ifndef THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES
#define THYME_ENVIRONMENTS_ROVER_DOMAIN_ROVER_TYPES

#include <rover_domain/utilities/shared_wrapper.hpp>

/*
*
* Agent abstraction
*
*/
namespace rover_domain {
class IRover;
using Agent = thyme::utilities::SharedWrap<IRover>;
}  // namespace rover_domain

#endif
