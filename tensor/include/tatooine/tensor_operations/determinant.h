#ifndef TATOOINE_TENSOR_OPERATIONS_DETERMINANT_H
#define TATOOINE_TENSOR_OPERATIONS_DETERMINANT_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T>
constexpr auto det(base_tensor<Tensor, T, 2, 2> const& A) -> T {
  return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T>
constexpr auto detAtA(base_tensor<Tensor, T, 2, 2> const& A) -> T {
  return A(0, 0) * A(0, 0) * A(1, 1) * A(1, 1) -
         2 * A(0, 0) * A(0, 1) * A(1, 0) * A(1, 1) +
         A(0, 1) * A(0, 1) * A(1, 0) * A(1, 0);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T>
constexpr auto det(base_tensor<Tensor, T, 3, 3> const& A) -> T {
  return (A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0)) * A(2, 2) +
         (A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2)) * A(2, 1) +
         (A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) * A(2, 0);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
