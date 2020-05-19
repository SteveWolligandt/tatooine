#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H
//==============================================================================
#include <complex>
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct is_unsigned_integral
    : std::integral_constant<bool,
                             std::is_integral_v<T> && std::is_unsigned_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_unsigned_integral_v = is_unsigned_integral<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
struct are_unsigned_integral;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_unsigned_integral_v =
    are_unsigned_integral<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_unsigned_integral<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_unsigned_integral<T>
    : std::integral_constant<bool, is_unsigned_integral_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_unsigned_integral<T0, T1, Ts...>
    : std::integral_constant<bool, are_unsigned_integral_v<T0> &&
                                       are_unsigned_integral_v<T1, Ts...>> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_unsigned_integral =
    std::enable_if_t<sizeof...(Ts) == 0 || are_unsigned_integral_v<Ts...>,
                     bool>;
//==============================================================================
template <typename... Ts>
struct are_floating_point;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_floating_point_v = are_floating_point<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_floating_point<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_floating_point<T>
    : std::integral_constant<bool, std::is_floating_point_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_floating_point<T0, T1, Ts...>
    : std::integral_constant<bool, are_floating_point_v<T0> &&
                                       are_floating_point_v<T1, Ts...>> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_floating_point =
    std::enable_if_t<sizeof...(Ts) == 0 || are_floating_point_v<Ts...>, bool>;
//==============================================================================
template <typename... Ts>
struct are_integral;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_integral_v = are_integral<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_integral<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_integral<T> : std::integral_constant<bool, std::is_integral_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_integral<T0, T1, Ts...>
    : std::integral_constant<bool, are_integral_v<T0> &&
                                       are_integral_v<T1, Ts...>> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_integral =
    std::enable_if_t<sizeof...(Ts) == 0 || are_integral_v<Ts...>, bool>;
//==============================================================================

template <typename... Ts>
struct are_arithmetic;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_arithmetic_v = are_arithmetic<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_arithmetic<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_arithmetic<T>
    : std::integral_constant<bool, std::is_arithmetic_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_arithmetic<T0, T1, Ts...>
    : std::integral_constant<bool, are_arithmetic_v<T0> &&
                                       are_arithmetic_v<T1, Ts...>> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_arithmetic =
    std::enable_if_t<sizeof...(Ts) == 0 || are_arithmetic_v<Ts...>, bool>;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_not_arithmetic =
    std::enable_if_t<sizeof...(Ts) == 0 || !are_arithmetic_v<Ts...>, bool>;

//==============================================================================
template <typename T>
struct is_complex : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_complex_v = is_complex<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
struct are_complex;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_complex_v = are_complex<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_complex<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_complex<T> : std::integral_constant<bool, is_complex_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_complex<T0, T1, Ts...>
    : std::integral_constant<bool,
                             are_complex_v<T0> && are_complex_v<T1, Ts...>> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_complex =
    std::enable_if_t<sizeof...(Ts) == 0 || are_complex_v<Ts...>, bool>;
//==============================================================================
template <typename... Ts>
struct are_floating_point_or_complex
    : std::integral_constant<bool, are_floating_point_v<Ts...> ||
                                       are_complex_v<Ts...>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_floating_point_or_complex_v =
    are_floating_point_or_complex<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
using enable_if_floating_point_or_complex = typename std::enable_if_t<
    sizeof...(Ts) == 0 || are_floating_point_or_complex_v<Ts...>, bool>;

//==============================================================================
template <typename... Ts>
struct are_arithmetic_or_complex
    : std::integral_constant<bool, are_arithmetic_v<Ts...> ||
                                       are_complex_v<Ts...>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_arithmetic_or_complex_v =
    are_arithmetic_or_complex<Ts...>::value;
//==============================================================================
template <typename... Ts>
using enable_if_arithmetic_or_complex = typename std::enable_if_t<
    sizeof...(Ts) == 0 || are_arithmetic_or_complex_v<Ts...>, bool>;
//==============================================================================
template <typename T, typename = void>
struct is_iterator : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_iterator_v = is_iterator<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_iterator<T, std::enable_if_t<!std::is_same_v<
                          typename std::iterator_traits<T>::value_type, void>>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
struct are_iterators;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_iterators_v = are_iterators<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_iterators<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_iterators<T> : std::integral_constant<bool, is_iterator<T>::value> {
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_iterators<T0, T1, Ts...>
    : std::integral_constant<bool, sizeof...(Ts) == 0 ||
                                       (are_iterators_v<T0> &&
                                        are_iterators_v<T1, Ts...>)> {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_iterator =
    std::enable_if_t<sizeof...(Ts) == 0 || are_iterators<Ts...>::value, bool>;
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
template <typename tensor_t, typename real_t, size_t... Dims>
struct base_tensor;
template <typename T>
struct is_vectorield : std::false_type {};
template <typename tensor_t, typename real_t, size_t N>
struct is_vectorield<base_tensor<tensor_t, real_t, N>> : std::true_type {};
template <typename T>
constexpr auto is_vectorield_v = is_vectorield<T>::value;
//==============================================================================
template <typename F, typename... Args>
using enable_if_invocable =
    std::enable_if_t<std::is_invocable_v<F, Args...>, bool>;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
