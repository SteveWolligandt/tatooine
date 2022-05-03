#ifndef TATOOINE_HAS_FIELDS_AVAILABLE_METHOD_H
#define TATOOINE_HAS_FIELDS_AVAILABLE_METHOD_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Void = void>
struct has_fields_available_method_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct has_fields_available_method_impl<
    T, decltype(std::declval<T>().fields_available)> : std::true_type {};
//==============================================================================
template <typename T>
constexpr auto has_fields_available_method() {
  return has_fields_available_method_impl<T>::value;
}
//------------------------------------------------------------------------------
template <typename T>
constexpr auto has_fields_available_method(T&& t) {
  return has_fields_available_method<T>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
