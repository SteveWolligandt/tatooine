#ifndef TATOOINE_TENSOR_UTILITY_H
#define TATOOINE_TENSOR_UTILITY_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// for comparison
template <typename Tensor0, typename Real0,
          typename Tensor1, typename Real1,
          size_t... Dims,
          std::enable_if_t<std::is_floating_point<Real0>::value ||
                           std::is_floating_point<Real1>::value,
                           bool> = true>
constexpr auto approx_equal(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                            const base_tensor<Tensor1, Real1, Dims...>& rhs,
                            promote_t<Real0, Real1> eps = 1e-6) {
  bool equal = true;
  lhs.for_indices([&](const auto... is) {
    if (std::abs(lhs(is...) - rhs(is...)) > eps) { equal = false; }
  });
  return equal;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
