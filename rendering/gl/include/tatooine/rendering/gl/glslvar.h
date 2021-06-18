#ifndef __YAVIN_GLSL_VAR_H__
#define __YAVIN_GLSL_VAR_H__

#include <string>

#include "dllexport.h"
#include "windowsundefines.h"

//==============================================================================
namespace yavin {
//==============================================================================

struct GLSLVar {
  enum modifier_t { UNIFORM, IN, OUT, UNKNOWN };
  modifier_t  modifier;
  std::string datatype;
  std::string name;

  DLL_API static auto modifier_to_string(const GLSLVar::modifier_t& modifier);
};

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
