#ifndef TATOOINE_RENDERING_GL_TESSELATIONCONTROLSHADER_H
#define TATOOINE_RENDERING_GL_TESSELATIONCONTROLSHADER_H
//==============================================================================
#include <string>

#include "dllexport.h"
#include "shaderstage.h"
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
class tesselationcontrolshader : public shaderstage {
 public:
  DLL_API tesselationcontrolshader(std::filesystem::path const& sourcepath);
  DLL_API tesselationcontrolshader(shadersource const& sourcepath);
  DLL_API tesselationcontrolshader(tesselationcontrolshader&& other);
};
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
