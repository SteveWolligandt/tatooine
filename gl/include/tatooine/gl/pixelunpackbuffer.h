#ifndef TATOOINE_GL_PIXEL_UNPACK_BUFFER_H
#define TATOOINE_GL_PIXEL_UNPACK_BUFFER_H

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
class pixelunpackbuffer : public buffer<GL_PIXEL_UNPACK_BUFFER, T> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using parent_type                         = buffer<GL_PIXEL_UNPACK_BUFFER, T>;
  using this_type                           = pixelunpackbuffer<T>;
  static constexpr usage_t default_usage = usage_t::STATIC_COPY;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  pixelunpackbuffer(usage_t usage = default_usage) : parent_type(usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(const pixelunpackbuffer& other) : parent_type(other) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(pixelunpackbuffer&& other) : parent_type(std::move(other)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(size_t n, usage_t usage = default_usage)
      : parent_type(n, usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(size_t n, const T& initial, usage_t usage = default_usage)
      : parent_type(n, initial, usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(const std::vector<T>& data, usage_t usage = default_usage)
      : parent_type(data, usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(std::initializer_list<T>&& list)
      : parent_type(std::move(list), default_usage) {}

  //----------------------------------------------------------------------------
  // assign operators
  //----------------------------------------------------------------------------
  auto& operator=(const pixelunpackbuffer& other) {
    parent_type::operator=(other);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& operator=(pixelunpackbuffer&& other) {
    parent_type::operator=(std::move(other));
    return *this;
  }
};

//==============================================================================
}  // namespace tatooine::gl
//==============================================================================

#endif
