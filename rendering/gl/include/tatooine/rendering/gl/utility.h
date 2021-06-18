#ifndef YAVIN_UTILITY_H
#define YAVIN_UTILITY_H
//==============================================================================
#include <array>
#include <concepts>
#include <utility>
#include "gltype.h"
//==============================================================================
namespace yavin {
//==============================================================================
/// creates an array of size N filled with val of type T
template <typename T, size_t N, size_t... Is>
auto make_array(const T& val, std::index_sequence<Is...>) {
  return std::array{((void)Is, val)...};
}
//------------------------------------------------------------------------------
/// creates an array of size N filled with val of type T
template <typename T, size_t N>
auto make_array(const T& val) {
  return make_array<T, N>(val, std::make_index_sequence<N>{});
}
//------------------------------------------------------------------------------
/// Applies function F to all elements of parameter pack ts
template <typename F, typename... Ts>
void for_each(F&& f, Ts&&... ts) {
  using discard_t = int[];
  // creates an array filled with zeros. while doing this f gets called with
  // elements of ts
  (void)discard_t{0, ((void)f(std::forward<Ts>(ts)), 0)...};
}
//==============================================================================
template <typename... Ts>
struct promote;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
using promote_t = typename promote<Ts...>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1>
struct promote<T0, T1> {
  using type =
      std::decay_t<decltype(true ? std::declval<T0>() : std::declval<T1>())>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename T2, typename... Ts>
struct promote<T0, T1, T2, Ts...> {
  using type = promote_t<T0, promote_t<T1, T2, Ts...>>;
};
//==============================================================================
template <typename T>
struct num_components;
template <typename T>
static constexpr auto num_components_v = num_components<T>::value;
template <typename T, size_t N>
struct num_components<std::array<T, N>> : std::integral_constant<size_t, N> {};
template <typename T> requires std::is_arithmetic_v<T>
struct num_components<T> : std::integral_constant<size_t, 1> {};
//==============================================================================
template <typename T>
struct value_type;
template <typename T>
static constexpr auto value_type_v = value_type<T>::value;
template <typename T, size_t N>
struct value_type<std::array<T, N>>
    : std::integral_constant<GLenum, gl_type_v<T>> {};
template <typename T> requires std::is_arithmetic_v<T>
struct value_type<T> : std::integral_constant<GLenum, gl_type_v<T>> {};
//==============================================================================
template <typename T, typename... Ts>
struct head {
  using type = T;
};
template <typename... Ts>
using head_t = typename head<Ts...>::type;
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
