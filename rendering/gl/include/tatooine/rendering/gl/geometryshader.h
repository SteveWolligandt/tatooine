#ifndef TATOOINE_RENDERING_GL_GEOMETRYSHADER_H
#define TATOOINE_RENDERING_GL_GEOMETRYSHADER_H
//==============================================================================
#include <string>

#include "dllexport.h"
#include "shaderstage.h"
//==============================================================================
namespace tatooine::rendering::gl {
//==============================================================================
class geometryshader : public shaderstage {
 public:
  DLL_API geometryshader(std::filesystem::path const& sourcepath);
  DLL_API geometryshader(shadersource const& sourcepath);
  DLL_API geometryshader(geometryshader&& other);
};
//==============================================================================
}  // namespace tatooine::rendering::gl
//==============================================================================
#endif
