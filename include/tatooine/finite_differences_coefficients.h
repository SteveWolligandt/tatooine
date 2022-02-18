#ifndef TATOOINE_FINITE_DIFFERENCES_COEFFICIENTS_H
#define TATOOINE_FINITE_DIFFERENCES_COEFFICIENTS_H
//==============================================================================
#include <tatooine/mat.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// read here for more information:
/// http://web.media.mit.edu/~crtaylor/calculator.html
template <floating_point... Xs>
auto finite_differences_coefficients(std::size_t const derivative_order,
                                     Xs... xs) {
  constexpr auto N    = sizeof...(xs);
  using real_type        = common_type<std::decay_t<Xs>...>;
  auto V              = mat<real_type, N, N>::vander(xs...);
  V                   = transposed(V);
  auto b              = vec<real_type, N>::zeros();
  b(derivative_order) = factorial(derivative_order);
  return solve(V, b);
}
//------------------------------------------------------------------------------
/// read here for more information:
/// http://web.media.mit.edu/~crtaylor/calculator.html
template <typename Tensor, floating_point T, size_t N>
auto finite_differences_coefficients(std::size_t const derivative_order,
                                     base_tensor<Tensor, T, N> const& v) {
  auto V              = mat<T, N, N>::vander(v);
  V                   = transposed(V);
  auto b              = vec<T, N>::zeros();
  b(derivative_order) = factorial(derivative_order);
  return solve(V, b);
}
//------------------------------------------------------------------------------
/// read here for more information:
/// http://web.media.mit.edu/~crtaylor/calculator.html
template <floating_point T>
auto finite_differences_coefficients(std::size_t const     derivative_order,
                                     std::vector<T> const& v) {
  auto const V        = transposed(tensor<T>::vander(v, v.size()));
  auto       b        = tensor<T>::zeros(v.size());
  b(derivative_order) = factorial(derivative_order);
  return solve(V, b).data();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
