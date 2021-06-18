#ifndef YAVIN_PIXEL_UNPACK_BUFFER_H
#define YAVIN_PIXEL_UNPACK_BUFFER_H

#include <initializer_list>
#include <iostream>
#include <vector>
#include "buffer.h"
#include "dllexport.h"
#include "errorcheck.h"
#include "glincludes.h"

//==============================================================================
namespace yavin {
//==============================================================================

template <typename T>
class pixelunpackbuffer : public buffer<GL_PIXEL_UNPACK_BUFFER, T> {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
  using parent_t                         = buffer<GL_PIXEL_UNPACK_BUFFER, T>;
  using this_t                           = pixelunpackbuffer<T>;
  static constexpr usage_t default_usage = usage_t::STATIC_COPY;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  pixelunpackbuffer(usage_t usage = default_usage) : parent_t(usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(const pixelunpackbuffer& other) : parent_t(other) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(pixelunpackbuffer&& other) : parent_t(std::move(other)) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(size_t n, usage_t usage = default_usage)
      : parent_t(n, usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(size_t n, const T& initial, usage_t usage = default_usage)
      : parent_t(n, initial, usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(const std::vector<T>& data, usage_t usage = default_usage)
      : parent_t(data, usage) {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  pixelunpackbuffer(std::initializer_list<T>&& list)
      : parent_t(std::move(list), default_usage) {}

  //----------------------------------------------------------------------------
  // assign operators
  //----------------------------------------------------------------------------
  auto& operator=(const pixelunpackbuffer& other) {
    parent_t::operator=(other);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto& operator=(pixelunpackbuffer&& other) {
    parent_t::operator=(std::move(other));
    return *this;
  }
};

//==============================================================================
}  // namespace yavin
//==============================================================================

#endif
