#ifndef TATOOINE_GL_GLSL_VAR_H
#define TATOOINE_GL_GLSL_VAR_H

#include <string>

#include "dllexport.h"
#include "windowsundefines.h"

//==============================================================================
namespace tatooine::gl {
//==============================================================================

struct GLSLVar {
  enum modifier_t { UNIFORM, IN, OUT, UNKNOWN };
  modifier_t  modifier;
  std::string datatype;
  std::string name;

  DLL_API static auto modifier_to_string(const GLSLVar::modifier_t& modifier);
};

//==============================================================================
}  // namespace tatooine::gl
//==============================================================================

#endif
