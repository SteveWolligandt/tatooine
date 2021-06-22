#ifndef TATOOINE_GL_VERTEXSHADER_H
#define TATOOINE_GL_VERTEXSHADER_H
//==============================================================================
#include <string>
#include "glincludes.h"

#include "shaderstage.h"
#include "dllexport.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class vertexshader : public shaderstage {
 public:
  DLL_API vertexshader(std::filesystem::path const& sourcepath);
  DLL_API vertexshader(shadersource const& sourcepath);
  DLL_API vertexshader(vertexshader&& other);
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
