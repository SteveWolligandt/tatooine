#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H
//==============================================================================
#include <type_traits>
#include <tatooine/is_complex.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename F, typename... Ts>
using is_predicate = std::is_same<bool, std::invoke_result_t<F, Ts...>>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename... Ts>
static constexpr inline auto is_predicate_v = is_predicate<F, Ts...>::value;
//==============================================================================
template <typename T>
static constexpr auto is_floating_point_v = std::is_floating_point_v<T>;
template <typename T>
static constexpr auto is_arithmetic_v = std::is_arithmetic_v<T>;
template <typename T>
static constexpr auto is_integral_v = std::is_integral_v<T>;
//==============================================================================
template <typename T>
struct num_components;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr size_t num_components_v = num_components<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point T>
struct num_components<T> : std::integral_constant<size_t, 1> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::integral T>
struct num_components<T> : std::integral_constant<size_t, 1> {};
//==============================================================================
template <typename T>
struct inner_value_type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
using inner_value_type_t = typename inner_value_type<T>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::integral T>
struct inner_value_type<T> {
  using type = T;
};
template <std::floating_point T>
struct inner_value_type<T> {
  using type = T;
};
//==============================================================================
template <typename... Ts>
struct promote;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
using promote_t = typename promote<Ts...>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct promote<T> {
  using type = std::decay_t<T>;
};
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
}  // namespace tatooine
//==============================================================================

#endif
