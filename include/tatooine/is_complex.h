#ifndef TATOOINE_IS_COMPLEX_H
#define TATOOINE_IS_COMPLEX_H
//==============================================================================
#include <complex>
#include <type_traits>
//==============================================================================
namespace tatooine {
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
}
//==============================================================================
#endif
