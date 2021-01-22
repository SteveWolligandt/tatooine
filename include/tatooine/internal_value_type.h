#ifndef TATOOINE_INTERNAL_VALUE_TYPE_H
#define TATOOINE_INTERNAL_VALUE_TYPE_H
//==============================================================================
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct internal_value_type_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename = void>
using internal_value_type = typename internal_value_type_impl<T>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
template <arithmetic T>
struct internal_value_type_impl<T> {
  using type = T;
};
#else
template <typename T>
struct internal_value_type_impl<T, enable_if_arithmetic<T>> {
  using type = T;
};
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
