#ifndef TATOOINE_GL_TESSELATIONEVALUATIONSHADER_H
#define TATOOINE_GL_TESSELATIONEVALUATIONSHADER_H
//==============================================================================
#include <string>

#include "dllexport.h"
#include "shaderstage.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class tesselationevaluationshader : public shaderstage {
 public:
  DLL_API tesselationevaluationshader(std::filesystem::path const& sourcepath);
  DLL_API tesselationevaluationshader(shadersource const& sourcepath);
  DLL_API tesselationevaluationshader(tesselationevaluationshader&& other);
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
