#ifndef TATOOINE_IS_TRANSPOSED_TENSOR_H
#define TATOOINE_IS_TRANSPOSED_TENSOR_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct is_transposed_tensor_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_transposed_tensor = is_transposed_tensor_impl<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
