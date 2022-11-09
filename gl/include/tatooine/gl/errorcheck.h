#ifndef TATOOINE_GL_ERROR_CHECK_H
#define TATOOINE_GL_ERROR_CHECK_H

#include "glincludes.h"
#include <stdexcept>
#include <string>
#include <string_view>
#include "dllexport.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class gl_error : public std::runtime_error {
 public:
  DLL_API gl_error(std::string_view const& function_name,
                   std::string_view const& message);
};
//==============================================================================
class gl_framebuffer_not_complete_error : public std::runtime_error {
 public:
  DLL_API explicit gl_framebuffer_not_complete_error(std::string const& what);
};
//==============================================================================
DLL_API auto gl_error_to_string(GLenum err) -> std::string;
DLL_API auto gl_framebuffer_error_to_string(GLenum status) -> std::string;
//==============================================================================
DLL_API auto gl_error_check(std::string_view function) -> void;
DLL_API auto gl_framebuffer_not_complete_check(GLuint fbo_id) -> void;
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
