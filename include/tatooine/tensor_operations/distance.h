#ifndef TATOOINE_TENSOR_OPERATIONS_DISTANCE_H
#define TATOOINE_TENSOR_OPERATIONS_DISTANCE_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t N>
constexpr auto squared_euclidean_distance(
    base_tensor<Tensor0, T0, N> const& lhs,
    base_tensor<Tensor1, T1, N> const& rhs) {
  return squared_euclidean_length(rhs - lhs);
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t N>
constexpr auto euclidean_distance(base_tensor<Tensor0, T0, N> const& lhs,
                                  base_tensor<Tensor1, T1, N> const& rhs) {
  return euclidean_length(rhs - lhs);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
