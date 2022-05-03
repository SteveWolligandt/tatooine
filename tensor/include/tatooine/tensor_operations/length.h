#ifndef TATOOINE_TENSOR_OPERATIONS_LENGTH_H
#define TATOOINE_TENSOR_OPERATIONS_LENGTH_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T, size_t N>
constexpr auto squared_euclidean_length(base_tensor<Tensor, T, N> const& t_in) {
  return dot(t_in, t_in);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto euclidean_length(base_tensor<Tensor, T, N> const& t_in) -> T {
  return std::sqrt(squared_euclidean_length(t_in));
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
