#ifndef TATOOINE_GL_TESSELATIONCONTROLSHADER_H
#define TATOOINE_GL_TESSELATIONCONTROLSHADER_H
//==============================================================================
#include <string>

#include "dllexport.h"
#include "shaderstage.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class tesselationcontrolshader : public shaderstage {
 public:
  DLL_API tesselationcontrolshader(std::filesystem::path const& sourcepath);
  DLL_API tesselationcontrolshader(shadersource const& sourcepath);
  DLL_API tesselationcontrolshader(tesselationcontrolshader&& other);
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
