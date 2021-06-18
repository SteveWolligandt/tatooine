#ifndef YAVIN_VBOHELPERS_H
#define YAVIN_VBOHELPERS_H

#include <array>
#include <initializer_list>
#include <iostream>
#include <vector>
#include "glincludes.h"
#include "gltype.h"
//==============================================================================
namespace yavin {
//==============================================================================
template <size_t num_attrs, class... Ts>
struct attr_offset;
//------------------------------------------------------------------------------
template <size_t num_attrs>
struct attr_offset<num_attrs> {
  constexpr static auto gen(size_t, size_t) {
    return std::array<size_t, num_attrs>();
  }
};
//------------------------------------------------------------------------------
template <size_t num_attrs, class FirstAttrib, class... RestAttribs>
struct attr_offset<num_attrs, FirstAttrib, RestAttribs...> {
  constexpr static auto gen(size_t off = 0, size_t idx = 0) {
    auto arr = attr_offset<num_attrs, RestAttribs...>::gen(
        off + sizeof(FirstAttrib), idx + 1);
    arr[idx] = off;
    return arr;
  }
};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
