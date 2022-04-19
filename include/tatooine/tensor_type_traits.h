#ifndef TATOOINE_TENSOR_TYPE_TRAITS_H
#define TATOOINE_TENSOR_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/tensor_concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct tensor_value_type_impl {
  using type = typename std::decay_t<T>::value_type;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex T, std::size_t N>
struct tensor_value_type_impl<std::array<T, N>> {
  using type = T;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic T>
struct tensor_value_type_impl<T> {
  using type = T;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <floating_point T>
struct tensor_value_type_impl<std::complex<T>> {
  using type = T;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
using tensor_value_type = typename tensor_value_type_impl<T>::type;
//==============================================================================
template <typename Tensor>
struct tensor_rank_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_tensor Tensor>
struct tensor_rank_impl<Tensor>
    : std::integral_constant<std::size_t, std::decay_t<Tensor>::rank()> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex Tensor>
struct tensor_rank_impl<Tensor> : std::integral_constant<std::size_t, 0> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex T, std::size_t N>
struct tensor_rank_impl<std::array<T, N>> : std::integral_constant<std::size_t, 1> {
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto tensor_rank = tensor_rank_impl<T>::value;
//==============================================================================
template <typename T>
struct tensor_num_components_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_tensor Tensor>
struct tensor_num_components_impl<Tensor>
    : std::integral_constant<std::size_t,
                             std::decay_t<Tensor>::num_components()> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex T>
struct tensor_num_components_impl<T> : std::integral_constant<std::size_t, 1> {
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex T, std::size_t N>
struct tensor_num_components_impl<std::array<T, N>>
    : std::integral_constant<std::size_t, N> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static auto constexpr tensor_num_components =
    tensor_num_components_impl<T>::value;
//==============================================================================
template <typename T>
struct tensor_dimensions_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_tensor Tensor>
struct tensor_dimensions_impl<Tensor> {
  static auto constexpr value = std::decay_t<Tensor>::dimensions();
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex T, std::size_t N>
struct tensor_dimensions_impl<std::array<T, N>> {
  static auto constexpr value = std::array{N};
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static auto constexpr tensor_dimensions = tensor_dimensions_impl<T>::value;
//==============================================================================
template <typename T, std::size_t I>
struct tensor_dimension_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_tensor Tensor, std::size_t I>
requires (I < tensor_rank<Tensor>)
struct tensor_dimension_impl<Tensor, I> {
  static auto constexpr value = std::decay_t<Tensor>::dimension(I);
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex T, std::size_t I>
struct tensor_dimension_impl<T, I> {
  static auto constexpr value = std::numeric_limits<T>::max();
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <arithmetic_or_complex T, std::size_t N>
struct tensor_dimension_impl<std::array<T, N>, 0> {
  static auto constexpr value = N;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, std::size_t I>
static auto constexpr tensor_dimension = tensor_dimension_impl<T, I>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
