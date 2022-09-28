#ifndef TATOOINE_FINITE_DIFFERENCES_COEFFICIENTS_H
#define TATOOINE_FINITE_DIFFERENCES_COEFFICIENTS_H
//==============================================================================
#include <tatooine/mat.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \addtogroup finite_differences_coefficients Finite Difference Coefficients
/// \{
//==============================================================================
/// See \ref fin_dif_what_is_this for an explanation.
auto finite_differences_coefficients(std::size_t const derivative_order,
                                     floating_point auto const... xs) {
  constexpr auto N    = sizeof...(xs);
  using real_type     = common_type<std::decay_t<decltype(xs)>...>;
  auto V              = mat<real_type, N, N>::vander(xs...);
  V                   = transposed(V);
  auto b              = vec<real_type, N>::zeros();
  b(derivative_order) =
      static_cast<real_type>(gcem::factorial(derivative_order));
  return *solve(V, b);
}
//------------------------------------------------------------------------------
/// See \ref fin_dif_what_is_this for an explanation.
template <typename Tensor, floating_point Real, std::size_t N>
auto finite_differences_coefficients(std::size_t const derivative_order,
                                     base_tensor<Tensor, Real, N> const& v) {
  auto V              = mat<Real, N, N>::vander(v);
  V                   = transposed(V);
  auto b              = vec<Real, N>::zeros();
  b(derivative_order) = static_cast<Real>(gcem::factorial<std::size_t>(derivative_order));
  return *solve(V, b);
}
//------------------------------------------------------------------------------
/// See \ref fin_dif_what_is_this for an explanation.
template <typename Tensor, floating_point Real, std::size_t N>
auto finite_differences_coefficients(std::size_t const derivative_order,
                                     vec<Real, N> const& v) {
  auto V              = mat<Real, N, N>::vander(v);
  V                   = transposed(V);
  auto b              = vec<Real, N>::zeros();
  b(derivative_order) = gcem::factorial(derivative_order);
  return *solve(V, b);
}
//------------------------------------------------------------------------------
/// See \ref fin_dif_what_is_this for an explanation.
template <floating_point_range R>
requires (!static_tensor<R>)
auto finite_differences_coefficients(std::size_t const derivative_order,
                                     R const& v) {
  using real_type     = typename std::decay_t<R>::value_type;
  auto const V        = transposed(tensor<real_type>::vander(v));
  auto       b        = tensor<real_type>::zeros(size(v));
  b(derivative_order) =
      static_cast<real_type>(gcem::factorial(derivative_order));
  return solve(V, b)->internal_container();
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
