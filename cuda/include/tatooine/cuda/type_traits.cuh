#ifndef TATOOINE_CUDA_TYPE_TRAITS_CUH
#define TATOOINE_CUDA_TYPE_TRAITS_CUH

#include <tatooine/type_traits.h>

//==============================================================================
namespace tatooine {
namespace cuda {
//==============================================================================

template <typename T>
struct is_freeable : std::false_type {};

//==============================================================================
template <typename... Ts>
struct are_freeable;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_freeable<T>
    : std::integral_constant<bool, is_freeable<T>::value> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_freeable<T0, T1, Ts...>
    : std::integral_constant<bool, are_freeable<T0>::value &&
                                   are_freeable<T1, Ts...>::value> {};
template <typename... Ts>
using enable_if_freeable =
    std::enable_if_t<are_freeable<Ts...>::value, bool>;

//==============================================================================
}  // namespace cuda
}  // namespace tatooine
//==============================================================================

#endif
