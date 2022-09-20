#ifndef TATOOINE_GL_COMPUTESHADER_H
#define TATOOINE_GL_COMPUTESHADER_H
//==============================================================================
#include <string>

#include "shaderstage.h"
#include "dllexport.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class computeshader : public shaderstage {
 public:
  computeshader(std::filesystem::path const& sourcepath);
  computeshader(shadersource const& sourcepath);
  computeshader(computeshader&& other);
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
