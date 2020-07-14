#ifndef TATOOINE_TENSOR_UTILITY_H
#define TATOOINE_TENSOR_UTILITY_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// for comparison
template <typename Tensor0, real_number Real0,
          typename Tensor1, real_number Real1,
          size_t... Dims>
constexpr auto approx_equal(const base_tensor<Tensor0, Real0, Dims...>& lhs,
                            const base_tensor<Tensor1, Real1, Dims...>& rhs,
                            promote_t<Real0, Real1> eps = 1e-6) {
  bool equal = true;
  lhs.for_indices([&](const auto... is) {
    if (std::abs(lhs(is...) - rhs(is...)) > eps) { equal = false; }
  });
  return equal;
}
//------------------------------------------------------------------------------
template <typename Tensor, real_number Real,
          size_t... Dims>
constexpr auto isnan(const base_tensor<Tensor, Real, Dims...>& t) -> bool {
  bool p = false;
  for_loop(
      [&](auto const... is) {
        if (std::isnan(t(is...))) {
          p = true;
          return false;
        }
        return true;
      },
      Dims...);
  return p;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
