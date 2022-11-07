#ifndef TATOOINE_TENSOR_UTILITY_H
#define TATOOINE_TENSOR_UTILITY_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// for comparison
template <static_tensor T0, static_tensor T1>
requires(same_dimensions<T0, T1>())
constexpr auto approx_equal(
    T0 const& lhs, T1 const& rhs,
    common_type<tatooine::value_type<T0>, tatooine::value_type<T1>> eps =
        1e-6) {
  auto equal = true;
  lhs.for_indices([&](const auto... is) {
    if (std::abs(lhs(is...) - rhs(is...)) > eps) {
      equal = false;
    }
  });
  return equal;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
constexpr auto isnan(base_tensor<Tensor, Real, Dims...> const& t) -> bool {
  auto p  = false;
  auto it = [&](auto const... is) {
    if (std::isnan(t(is...))) {
      p = true;
      return false;
    }
    return true;
  };
  for_loop(it, Dims...);
  return p;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
