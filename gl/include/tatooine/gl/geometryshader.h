#ifndef TATOOINE_GL_GEOMETRYSHADER_H
#define TATOOINE_GL_GEOMETRYSHADER_H
//==============================================================================
#include <string>

#include "dllexport.h"
#include "shaderstage.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class geometryshader : public shaderstage {
 public:
  DLL_API geometryshader(std::filesystem::path const& sourcepath);
  DLL_API geometryshader(shadersource const& sourcepath);
  DLL_API geometryshader(geometryshader&& other);
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
