#ifndef TATOOINE_GL_ATOMIC_COUNTER_BUFFER_H
#define TATOOINE_GL_ATOMIC_COUNTER_BUFFER_H
//==============================================================================
#include <initializer_list>
#include <iostream>
#include <vector>
#include <tatooine/gl/buffer.h>
#include <tatooine/gl/dllexport.h>
#include <tatooine/gl/errorcheck.h>
//==============================================================================
namespace tatooine::gl {
//==============================================================================
class atomiccounterbuffer
    : public buffer<GL_ATOMIC_COUNTER_BUFFER, GLuint> {
 public:
  using parent_type = buffer<GL_ATOMIC_COUNTER_BUFFER, GLuint>;
  using this_type   = atomiccounterbuffer;
  static const usage_t default_usage = DYNAMIC_DRAW;

  DLL_API explicit atomiccounterbuffer(usage_t usage = default_usage);

  atomiccounterbuffer(const this_type& other)     = default;
  atomiccounterbuffer(this_type&& other) noexcept = default;

  auto operator=(const this_type& other) -> atomiccounterbuffer&     = default;
  auto operator=(this_type&& other) noexcept -> atomiccounterbuffer& = default;

  ~atomiccounterbuffer() = default;

  DLL_API explicit atomiccounterbuffer(GLsizei n, usage_t usage = default_usage);
  DLL_API atomiccounterbuffer(GLsizei n, GLuint initial,
                              usage_t usage = default_usage);
  DLL_API explicit atomiccounterbuffer(const std::vector<GLuint>& data,
                              usage_t usage = default_usage);
  DLL_API atomiccounterbuffer(std::initializer_list<GLuint>&& data);

  DLL_API auto        set_all_to(GLuint val) -> void;
  auto                to_zero() { set_all_to(0); }
  DLL_API auto        bind(GLuint i) const -> void;
  DLL_API static auto unbind(GLuint i) -> void;
};
//==============================================================================
}  // namespace tatooine::gl
//==============================================================================
#endif
