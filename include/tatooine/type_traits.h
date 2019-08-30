#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H

#include <complex>
#include <type_traits>

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
template <typename... Ts>
using enable_if_integral =
    typename std::enable_if_t<(std::is_integral_v<Ts> && ...)>;

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
