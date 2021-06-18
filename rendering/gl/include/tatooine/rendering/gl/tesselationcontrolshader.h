#ifndef YAVIN_TESSELATIONCONTROLSHADER_H
#define YAVIN_TESSELATIONCONTROLSHADER_H
//==============================================================================
#include <string>

#include "dllexport.h"
#include "shaderstage.h"
//==============================================================================
namespace yavin {
//==============================================================================
class tesselationcontrolshader : public shaderstage {
 public:
  DLL_API tesselationcontrolshader(std::filesystem::path const& sourcepath);
  DLL_API tesselationcontrolshader(shadersource const& sourcepath);
  DLL_API tesselationcontrolshader(tesselationcontrolshader&& other);
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
