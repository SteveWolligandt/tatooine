#ifndef TATOOINE_RENDERING_GL_MUTEX_HANDLER_H
#define TATOOINE_RENDERING_GL_MUTEX_HANDLER_H
//==============================================================================
#include <mutex>
//==============================================================================
namespace tatooine::rendering::gl::detail {
//==============================================================================
struct mutex {
  static std::mutex buffer;
  static std::mutex gl_call;
};
//==============================================================================
}  // namespace tatooine::rendering::gl::detail
//==============================================================================
#endif
