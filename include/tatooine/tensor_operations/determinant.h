#ifndef TATOOINE_TENSOR_OPERATIONS_DETERMINANT_H
#define TATOOINE_TENSOR_OPERATIONS_DETERMINANT_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T>
constexpr auto det(base_tensor<Tensor, T, 2, 2> const& m) -> T {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T>
constexpr auto detAtA(base_tensor<Tensor, T, 2, 2> const& m) -> T {
  return m(0, 0) * m(0, 0) * m(1, 1) * m(1, 1) +
         m(0, 1) * m(0, 1) * m(1, 0) * m(1, 0) -
         2 * m(0, 0) * m(1, 0) * m(0, 1) * m(1, 1);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T>
constexpr auto det(base_tensor<Tensor, T, 3, 3> const& m) -> T {
  return m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
         m(0, 2) * m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1) * m(0, 2) -
         m(2, 1) * m(1, 2) * m(0, 0) - m(2, 2) * m(1, 0) * m(0, 2);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
