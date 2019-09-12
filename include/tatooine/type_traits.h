#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H

#include <complex>
#include <type_traits>
#include <ginac/ginac.h>

//==============================================================================
namespace tatooine {
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
template <typename T>
struct is_symbolic
    : std::integral_constant<bool, std::is_same_v<T, GiNaC::ex> ||
                                   std::is_same_v<T, GiNaC::symbol>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_symbolic_v = is_symbolic<T>::value;

//==============================================================================
template <typename T>
struct is_arithmetic_or_symbolic
    : std::integral_constant<bool,
                             std::is_arithmetic_v<T> || is_symbolic_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_arithmetic_or_symbolic_v =
    is_arithmetic_or_symbolic<T>::value;

//==============================================================================
template <typename T>
struct is_complex : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_complex_v = is_complex<T>::value;

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
template <typename... Ts>
using enable_if_integral =
    typename std::enable_if_t<(std::is_integral_v<Ts> && ...)>;
template <typename... Ts>
using enable_if_not_integral =
    typename std::enable_if_t<(!std::is_integral_v<Ts> && ...)>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_floating_point =
    typename std::enable_if_t<(std::is_floating_point_v<Ts> && ...)>;
template <typename... Ts>
using enable_if_not_floating_point =
    typename std::enable_if_t<(!std::is_floating_point_v<Ts> && ...)>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_arithmetic =
    typename std::enable_if_t<(std::is_arithmetic_v<Ts> && ...)>;
template <typename... Ts>
using enable_if_not_arithmetic =
    typename std::enable_if_t<(!std::is_arithmetic_v<Ts> && ...)>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_symbolic =
    typename std::enable_if_t<(is_symbolic_v<Ts> && ...)>;
template <typename... Ts>
using enable_if_not_symbolic =
    typename std::enable_if_t<(!is_symbolic_v<Ts> && ...)>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_arithmetic_or_symbolic =
    typename std::enable_if_t<(is_arithmetic_or_symbolic_v<Ts> && ...)>;

//==============================================================================
#define make_sfinae_test(name, method)                                     \
  template <typename T>                                                    \
  struct name {                                                            \
    template <typename S>                                                  \
    static char test(decltype(&S::method));                                \
    template <typename S>                                                  \
    static long           test(...);                                       \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(char);      \
    constexpr             operator bool() const noexcept { return value; } \
    constexpr auto        operator()() const noexcept { return value; }    \
  };                                                                       \
                                                                           \
  template <typename T>                                                    \
  constexpr auto name##_v = name<T>{}

//==============================================================================
//! SFINAE test if is_domain function exists
make_sfinae_test(has_in_domain, in_domain);
#undef make_sfinae_test

//==============================================================================
template <typename tensor_t, typename real_t, size_t... Dims>
struct base_tensor;
template <typename T>
struct is_vectorield : std::false_type {};
template <typename tensor_t, typename real_t, size_t N>
struct is_vectorield<base_tensor<tensor_t, real_t, N>> : std::true_type {};
template <typename T>
inline constexpr auto is_vectorield_v = is_vectorield<T>::value;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
