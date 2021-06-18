#ifndef YAVIN_ATOMIC_COUNTER_BUFFER_H
#define YAVIN_ATOMIC_COUNTER_BUFFER_H
//==============================================================================
#include <initializer_list>
#include <iostream>
#include <vector>
#include <yavin/buffer.h>
#include <yavin/dllexport.h>
#include <yavin/errorcheck.h>
//==============================================================================
namespace yavin {
//==============================================================================
class atomiccounterbuffer
    : public buffer<GL_ATOMIC_COUNTER_BUFFER, GLuint> {
 public:
  using parent_t = buffer<GL_ATOMIC_COUNTER_BUFFER, GLuint>;
  using this_t   = atomiccounterbuffer;
  static const usage_t default_usage = DYNAMIC_DRAW;

  DLL_API explicit atomiccounterbuffer(usage_t usage = default_usage);

  atomiccounterbuffer(const this_t& other)     = default;
  atomiccounterbuffer(this_t&& other) noexcept = default;

  auto operator=(const this_t& other) -> atomiccounterbuffer&     = default;
  auto operator=(this_t&& other) noexcept -> atomiccounterbuffer& = default;

  ~atomiccounterbuffer() = default;

  DLL_API explicit atomiccounterbuffer(size_t n, usage_t usage = default_usage);
  DLL_API atomiccounterbuffer(size_t n, GLuint initial,
                              usage_t usage = default_usage);
  DLL_API explicit atomiccounterbuffer(const std::vector<GLuint>& data,
                              usage_t usage = default_usage);
  DLL_API atomiccounterbuffer(std::initializer_list<GLuint>&& data);

  DLL_API void        set_all_to(GLuint val);
  void                to_zero() { set_all_to(0); }
  DLL_API void        bind(size_t i) const;
  DLL_API static void unbind(size_t i);
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
