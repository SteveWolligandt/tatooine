#ifndef TATOOINE_CXX14_TYPE_TRAITS_H
#define TATOOINE_CXX14_TYPE_TRAITS_H

#include <type_traits>

//==============================================================================
namespace tatooine {
namespace cxx14 {
//==============================================================================

template <typename... Ts>
struct are_integral;
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
using enable_if_arithmetic = std::enable_if_t<are_arithmetic<Ts...>::value>;

//==============================================================================
}  // namespace cxx14
}  // namespace tatooine
//==============================================================================

#endif
