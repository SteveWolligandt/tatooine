#ifndef TATOOINE_NUM_COMPONENTS_H
#define TATOOINE_NUM_COMPONENTS_H
//==============================================================================
#include <tatooine/type_traits.h>

#include <array>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename = void>
struct num_components_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto num_components = num_components_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
requires is_arithmetic<T> struct num_components_impl<T>
    : std::integral_constant<size_t, 1> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N>
struct num_components_impl<std::array<T, N>>
    : std::integral_constant<size_t, N> {};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
