#ifndef TATOOINE_NUM_COMPONENTS_H
#define TATOOINE_NUM_COMPONENTS_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename = void>
struct num_components_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto num_components = num_components_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
template <typename T>
requires is_arithmetic<T> struct num_components_impl<T>
    : std::integral_constant<size_t, 1> {};
#else
template <typename T>
struct num_components_impl<T, enable_if_arithmetic<T> >
    : std::integral_constant<size_t, 1> {};
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
