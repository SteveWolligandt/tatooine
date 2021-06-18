#ifndef YAVIN_GEOMETRYSHADER_H
#define YAVIN_GEOMETRYSHADER_H
//==============================================================================
#include <string>

#include "dllexport.h"
#include "shaderstage.h"
//==============================================================================
namespace yavin {
//==============================================================================
class geometryshader : public shaderstage {
 public:
  DLL_API geometryshader(std::filesystem::path const& sourcepath);
  DLL_API geometryshader(shadersource const& sourcepath);
  DLL_API geometryshader(geometryshader&& other);
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
