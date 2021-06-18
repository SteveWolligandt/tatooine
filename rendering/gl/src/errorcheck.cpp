#include <yavin/ansiformat.h>
#include <yavin/errorcheck.h>
#include <iostream>
#include <sstream>

#include <yavin/glfunctions.h>
//==============================================================================
namespace yavin {
//==============================================================================
gl_error::gl_error(const std::string& function_name, const std::string& message)
    : std::runtime_error(std::string{ansi::red} + std::string{ansi::bold} +
                         "[" + function_name + "] " + std::string{ansi::reset} +
                         message) {}
//------------------------------------------------------------------------------
gl_framebuffer_not_complete_error::gl_framebuffer_not_complete_error(
    const std::string& what)
    : std::runtime_error("[FrameBuffer incomplete] " + what) {}
//------------------------------------------------------------------------------
auto gl_error_to_string(GLenum err) -> std::string{
  switch (err) {
    case GL_INVALID_ENUM:      return "invalid enum";
    case GL_INVALID_VALUE:     return "invalid value";
    case GL_INVALID_OPERATION: return "invalid operation";
    case GL_STACK_OVERFLOW:    return "stack overflow";
    case GL_STACK_UNDERFLOW:   return "stack underflow";
    case GL_OUT_OF_MEMORY:     return "out of memory";
    case GL_CONTEXT_LOST:      return "context lost";
    case GL_TABLE_TOO_LARGE:   return "tablet too large";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "invalid framebuffer operation";
    default: return "unknown GL error";
  }
}
//------------------------------------------------------------------------------
auto gl_framebuffer_error_to_string(GLenum status) -> std::string{
  switch (status) {
    case GL_FRAMEBUFFER_UNDEFINED: return "undefined";
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: return "incomplete attachment";
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
      return "incomplete missing attachment";
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: return "incomplete draw buffer";
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: return "incomplete read buffer";
    case GL_FRAMEBUFFER_UNSUPPORTED: return "unsupported";
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: return "incomplete multisample";
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
      return "incomplete layer targets";
  }
  return "";
}
//------------------------------------------------------------------------------
void gl_error_check(std::string_view function) {
  const auto err = gl::get_error();
  if (err != GL_NO_ERROR) {
    const auto err_str = gl_error_to_string(err);
    throw gl_error(std::string{function}, err_str);
  }
}
//------------------------------------------------------------------------------
void gl_framebuffer_not_complete_check(const GLuint fbo_id) {
  const auto status =
      gl::check_named_framebuffer_status(fbo_id, GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    const auto status_str = gl_framebuffer_error_to_string(status);
    throw gl_framebuffer_not_complete_error(status_str);
  }
}

//==============================================================================
}  // namespace yavin
//==============================================================================
