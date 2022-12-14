#include <tatooine/gl/glslvar.h>

//==============================================================================
namespace tatooine::gl {
//==============================================================================

auto GLSLVar::modifier_to_string(const GLSLVar::modifier_t& modifier) {
  switch (modifier) {
    case GLSLVar::UNIFORM: return "uniform";
    case GLSLVar::IN: return "in";
    case GLSLVar::OUT: return "out";
    default: return "";
  }
}

//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
