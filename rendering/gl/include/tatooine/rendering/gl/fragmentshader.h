#ifndef YAVIN_FRAGMENTSHADER_H
#define YAVIN_FRAGMENTSHADER_H
//==============================================================================
#include <string>
#include "glincludes.h"

#include "shaderstage.h"
#include "dllexport.h"
//==============================================================================
namespace yavin {
//==============================================================================
class fragmentshader : public shaderstage {
 public:
  DLL_API fragmentshader(std::filesystem::path const& sourcepath);
  DLL_API fragmentshader(shadersource const& sourcepath);
  DLL_API fragmentshader(fragmentshader&& other);
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
