#ifndef TATOOINE_HAS_TRIANGLE_AT_METHOD_H
#define TATOOINE_HAS_TRIANGLE_AT_METHOD_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Void = void>
struct has_triangle_at_method_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
struct has_triangle_at_method_impl<
    T, std::void_t<decltype(std::declval<T>().triangle_at(size_t{}))>>
    : std::true_type {};
//==============================================================================
template <typename T>
constexpr auto has_triangle_at_method() {
  return has_triangle_at_method_impl<T>::value;
}
//------------------------------------------------------------------------------
template <typename T>
constexpr auto has_triangle_at_method(T&& /*t*/) {
  return has_triangle_at_method<T>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
