#ifndef TATOOINE_TENSOR_OPERATIONS_CROSS_H
#define TATOOINE_TENSOR_OPERATIONS_CROSS_H
//==============================================================================
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor0, typename T0, typename Tensor1, typename T1>
constexpr auto cross(base_tensor<Tensor0, T0, 3> const& lhs,
                     base_tensor<Tensor1, T1, 3> const& rhs) {
  return vec<common_type<T0, T1>, 3>{lhs(1) * rhs(2) - lhs(2) * rhs(1),
                                     lhs(2) * rhs(0) - lhs(0) * rhs(2),
                                     lhs(0) * rhs(1) - lhs(1) * rhs(0)};
}
//==============================================================================
template <typename T0, typename T1>
constexpr auto cross(tensor<T0> const& lhs,
                     tensor<T1> const& rhs) {
  return tensor<common_type<T0, T1>>{lhs(1) * rhs(2) - lhs(2) * rhs(1),
                                     lhs(2) * rhs(0) - lhs(0) * rhs(2),
                                     lhs(0) * rhs(1) - lhs(1) * rhs(0)};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
