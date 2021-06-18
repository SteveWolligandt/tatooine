#ifndef YAVIN_COMPUTESHADER_H
#define YAVIN_COMPUTESHADER_H
//==============================================================================
#include <string>

#include "shaderstage.h"
#include "dllexport.h"
//==============================================================================
namespace yavin {
//==============================================================================
class computeshader : public shaderstage {
 public:
  DLL_API computeshader(std::filesystem::path const& sourcepath);
  DLL_API computeshader(shadersource const& sourcepath);
  DLL_API computeshader(computeshader&& other);
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
