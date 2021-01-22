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
#ifdef __cpp_concepts
template <arithmetic T>
struct internal_value_type_impl<T> {
  using type = T;
};
#else
template <>
struct internal_value_type_impl<bool> {
  using type = bool;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<char> {
  using type = char;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<char16_t> {
  using type = char16_t;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<char32_t> {
  using type = char32_t;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<wchar_t> {
  using type = wchar_t;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<short> {
  using type = short;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<int> {
  using type = int;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<long> {
  using type = long;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<long long> {
  using type = long long;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<unsigned char> {
  using type = unsigned char;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<unsigned short> {
  using type = unsigned short;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<unsigned int> {
  using type = unsigned int;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<unsigned long> {
  using type = unsigned long;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct internal_value_type_impl<unsigned long long> {
  using type = unsigned long long;
};
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
