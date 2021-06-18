#ifndef YAVIN_ERROR_CHECK_H
#define YAVIN_ERROR_CHECK_H

#include "glincludes.h"
#include <stdexcept>
#include <string>
#include "compilation.h"
#include "dllexport.h"

//==============================================================================
namespace yavin {
//==============================================================================

class gl_error : public std::runtime_error {
 public:
  DLL_API gl_error(const std::string& function_name,
                   const std::string& message);
};

class gl_framebuffer_not_complete_error : public std::runtime_error {
 public:
  DLL_API explicit gl_framebuffer_not_complete_error(const std::string& what);
};

DLL_API auto gl_error_to_string(GLenum err) -> std::string;
DLL_API auto gl_framebuffer_error_to_string(GLenum status) -> std::string;

DLL_API void gl_error_check(std::string_view function);
DLL_API void gl_framebuffer_not_complete_check(GLuint fbo_id);
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
