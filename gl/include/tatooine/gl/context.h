#ifndef TATOOINE_GL_CONTEXT_H
#define TATOOINE_GL_CONTEXT_H
//==============================================================================
#include <tatooine/gl/glfw/context.h>
#include <tatooine/gl/glincludes.h>

#include <array>
#include <memory>
#include <iostream>
#include <list>
#include <string>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class window;
class context {
 public:
  //============================================================================
  // members
  //============================================================================
  std::unique_ptr<glfw::context> m_glfw_context;

  //============================================================================
  // ctors / dtor
  //============================================================================
 public:
  context();
  context(context&&) noexcept = default;
  auto operator=(context&&) noexcept -> context& = default;
  ~context()                                     = default;
  context(context& parent);
  context(window& parent);

  //============================================================================
  // methods
  //============================================================================
 public:
  context create_shared_context();
  void    make_current();
  void    release();
  auto    get() -> auto& { return *m_glfw_context; }
  auto    get() const -> auto const& { return *m_glfw_context; }
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
