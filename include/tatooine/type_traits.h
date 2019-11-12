#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H

#include "cxxstd.h"

#if has_cxx17_support()
#include <ginac/ginac.h>
#endif

#include <complex>
#include <type_traits>

//==============================================================================
namespace tatooine {
//==============================================================================
#if has_cxx17_support()
template <typename T>
struct is_symbolic
    : std::integral_constant<bool, std::is_same<T, GiNaC::ex>::value ||
                                       std::is_same<T, GiNaC::symbol>::value> {
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_symbolic_v = is_symbolic<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
struct are_symbolic;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_symbolic<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_symbolic<T> : std::integral_constant<bool, is_symbolic<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_symbolic<T0, T1, Ts...>
    : std::integral_constant<bool, are_symbolic<T0>::value &&
                                       are_symbolic<T1, Ts...>::value> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_symbolic = std::enable_if_t<are_symbolic<Ts...>::value, bool>;
#endif

//==============================================================================
template <typename... Ts>
struct are_floating_point;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_floating_point<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_floating_point<T>
    : std::integral_constant<bool, std::is_floating_point<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_floating_point<T0, T1, Ts...>
    : std::integral_constant<bool, are_floating_point<T0>::value &&
                                       are_floating_point<T1, Ts...>::value> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_floating_point =
    std::enable_if_t<are_floating_point<Ts...>::value, bool>;
//==============================================================================
template <typename... Ts>
struct are_integral;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_integral<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_integral<T>
    : std::integral_constant<bool, std::is_integral<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_integral<T0, T1, Ts...>
    : std::integral_constant<bool, are_integral<T0>::value &&
                                       are_integral<T1, Ts...>::value> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_integral = std::enable_if_t<are_integral<Ts...>::value, bool>;
//==============================================================================

template <typename... Ts>
struct are_arithmetic;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_arithmetic<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_arithmetic<T>
    : std::integral_constant<bool, std::is_arithmetic<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_arithmetic<T0, T1, Ts...>
    : std::integral_constant<bool, are_arithmetic<T0>::value &&
                                       are_arithmetic<T1, Ts...>::value> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_arithmetic =
    std::enable_if_t<are_arithmetic<Ts...>::value, bool>;

//==============================================================================
template <typename T>
struct is_complex : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
struct are_complex;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_complex<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_complex<T>
    : std::integral_constant<bool, is_complex<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_complex<T0, T1, Ts...>
    : std::integral_constant<bool, are_complex<T0>::value &&
                                       are_complex<T1, Ts...>::value> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_complex = std::enable_if_t<are_complex<Ts...>::value, bool>;

//==============================================================================
template <typename... Ts>
struct are_arithmetic_or_complex
    : std::integral_constant<bool, are_arithmetic<Ts...>::value ||
                                   are_complex<Ts...>::value> {};
//==============================================================================
template <typename... Ts>
using enable_if_arithmetic_or_complex =
    typename std::enable_if_t<are_arithmetic_or_complex<Ts...>::value, bool>;
//==============================================================================
#if has_cxx17_support()
template <typename... Ts>
struct are_arithmetic_or_symbolic
    : std::integral_constant<bool, are_arithmetic<Ts...>::value ||
                                   are_symbolic<Ts...>::value> {};
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_arithmetic_or_symbolic =
    typename std::enable_if_t<are_arithmetic_or_symbolic<Ts...>::value, bool>;
//==============================================================================
template <typename... Ts>
struct are_arithmetic_complex_or_symbolic
    : std::integral_constant<bool, are_arithmetic<Ts...>::value ||
                                   are_complex<Ts...>::value ||
                                   are_symbolic<Ts...>::value> {};
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_arithmetic_complex_or_symbolic =
    typename std::enable_if_t<are_arithmetic_complex_or_symbolic<Ts...>::value,
                              bool>;
#endif
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
    static char test(decltype(&S::method));                                \
    template <typename S>                                                  \
    static long           test(...);                                       \
    static constexpr bool value = sizeof(test<T>(0)) == sizeof(char);      \
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
template <typename tensor_t, typename real_t, size_t... Dims>
struct base_tensor;
template <typename T>
struct is_vectorield : std::false_type {};
template <typename tensor_t, typename real_t, size_t N>
struct is_vectorield<base_tensor<tensor_t, real_t, N>> : std::true_type {};
template <typename T>
constexpr auto is_vectorield_v = is_vectorield<T>::value;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
