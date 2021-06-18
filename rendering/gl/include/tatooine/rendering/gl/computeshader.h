#ifndef TATOOINE_RENDERING_GL_COMPUTESHADER_H
#define TATOOINE_RENDERING_GL_COMPUTESHADER_H
//==============================================================================
#include <string>

#include "shaderstage.h"
#include "dllexport.h"
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
class computeshader : public shaderstage {
 public:
  DLL_API computeshader(std::filesystem::path const& sourcepath);
  DLL_API computeshader(shadersource const& sourcepath);
  DLL_API computeshader(computeshader&& other);
};
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
