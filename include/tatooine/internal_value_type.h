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
template <typename T>
using internal_value_type = typename internal_value_type_impl<T>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic T>
struct internal_value_type_impl<T> {
  using type = T;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
