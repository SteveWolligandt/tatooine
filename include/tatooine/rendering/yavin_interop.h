#ifndef TATOOINE_RENDERING_YAVIN_INTEROP_H
#define TATOOINE_RENDERING_YAVIN_INTEROP_H
//==============================================================================
#include <tatooine/vec.h>
#include <yavin/utility.h>
//==============================================================================
namespace yavin {
//==============================================================================
template <typename T, size_t N>
struct num_components<tatooine::vec<T, N>>
    : std::integral_constant<size_t, N> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N>
struct num_components<tatooine::tensor<T, N>>
    : std::integral_constant<size_t, N> {};
//==============================================================================
template <typename T, size_t N>
struct value_type<tatooine::vec<T, N>>
    : std::integral_constant<GLenum, gl_type_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t N>
struct value_type<tatooine::tensor<T, N>>
    : std::integral_constant<GLenum, gl_type_v<T>> {};
//==============================================================================
}  // namespace yavin
//==============================================================================
#endif
