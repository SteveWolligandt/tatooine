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
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_integral =
    std::enable_if_t<(std::is_integral_v<Ts> && ...), bool>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_signed_integral = std::enable_if_t<
    (std::is_integral_v<Ts> && ...) && (std::is_signed_v<Ts> && ...), bool>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_unsigned_integral = std::enable_if_t<
    (std::is_integral_v<Ts> && ...) && !(std::is_signed_v<Ts> && ...), bool>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_floating_point =
    std::enable_if_t<(std::is_floating_point_v<Ts> && ...), bool>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_arithmetic =
    std::enable_if_t<(std::is_arithmetic_v<Ts> && ...), bool>;
//==============================================================================
template <typename T>
struct num_components;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr size_t num_components_v = num_components<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct num_components<double> : std::integral_constant<size_t, 1> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct num_components<float> : std::integral_constant<size_t, 1> {};
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
#define make_sfinae_test(name, method)                                     \
  template <typename T>                                                    \
  struct name {                                                            \
    template <typename S>                                                  \
    static auto test(decltype(&S::method)) -> char;                        \
    template <typename S>                                                  \
    static auto test(...) -> long;                                         \
                                                                           \
    static constexpr auto value = sizeof(test<T>(0)) == sizeof(char);      \
    constexpr             operator bool() const noexcept { return value; } \
    constexpr auto        operator()() const noexcept { return value; }    \
  };                                                                       \
                                                                           \
  template <typename T>                                                    \
  constexpr auto name##_v = name<T> {}

//==============================================================================
//! SFINAE test if is_domain function exists
make_sfinae_test(has_in_domain, in_domain);
#undef make_sfinae_test
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
