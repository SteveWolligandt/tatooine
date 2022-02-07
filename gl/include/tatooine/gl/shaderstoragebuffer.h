#ifndef TATOOINE_GL_SHADER_STORAGE_BUFFER_H
#define TATOOINE_GL_SHADER_STORAGE_BUFFER_H

#include <initializer_list>
#include <iostream>
#include <vector>
#include "buffer.h"
#include "dllexport.h"
#include "errorcheck.h"
#include "glincludes.h"
//==============================================================================
namespace tatooine::gl {
//==============================================================================
template <typename T>
class shaderstoragebuffer : public buffer<GL_SHADER_STORAGE_BUFFER, T> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using parent_type                     = buffer<GL_SHADER_STORAGE_BUFFER, T>;
  static const usage_t default_usage = usage_t::DYNAMIC_DRAW;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  shaderstoragebuffer(usage_t usage = default_usage) : parent_type(usage) {}
  //----------------------------------------------------------------------------
  shaderstoragebuffer(const shaderstoragebuffer& other) : parent_type(other) {}
  //----------------------------------------------------------------------------
  shaderstoragebuffer(shaderstoragebuffer&& other)
      : parent_type(std::move(other)) {}
  //----------------------------------------------------------------------------
  shaderstoragebuffer(size_t n, usage_t usage = default_usage)
      : parent_type(n, usage) {}
  //----------------------------------------------------------------------------
  shaderstoragebuffer(size_t n, const T& initial, usage_t usage = default_usage)
      : parent_type(n, initial, usage) {}
  //----------------------------------------------------------------------------
  shaderstoragebuffer(const std::vector<T>& data, usage_t usage = default_usage)
      : parent_type(data, usage) {}
  //----------------------------------------------------------------------------
  shaderstoragebuffer(std::initializer_list<T>&& list)
      : parent_type(std::move(list), default_usage) {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  void bind(GLuint index) const {
    gl::bind_buffer_base(GL_SHADER_STORAGE_BUFFER, index, this->id());
  }
  //----------------------------------------------------------------------------
  static void unbind(size_t index) {
    gl::bind_buffer_base(GL_SHADER_STORAGE_BUFFER, index, 0);
  }
};

//==============================================================================
}  // namespace tatooine::gl
//==============================================================================

#endif
