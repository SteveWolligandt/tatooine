#ifndef YAVIN_VERTEXSHADER_H
#define YAVIN_VERTEXSHADER_H
//==============================================================================
#include <string>
#include "glincludes.h"

#include "shaderstage.h"
#include "dllexport.h"
//==============================================================================
namespace yavin {
//==============================================================================
class vertexshader : public shaderstage {
 public:
  DLL_API vertexshader(std::filesystem::path const& sourcepath);
  DLL_API vertexshader(shadersource const& sourcepath);
  DLL_API vertexshader(vertexshader&& other);
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
