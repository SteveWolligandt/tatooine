#ifndef TATOOINE_FINITE_DIFFERENCES_COEFFICIENTS_H
#define TATOOINE_FINITE_DIFFERENCES_COEFFICIENTS_H
//==============================================================================
#include <tatooine/mat.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// read her for more information:
/// http://web.media.mit.edu/~crtaylor/calculator.html
template <floating_point... Xs>
auto finite_differences_coefficients(std::size_t d, Xs... xs) {
  constexpr auto N = sizeof...(xs);
  using real_t     = promote_t<std::decay_t<Xs>...>;
  auto V           = mat<real_t, N, N>::vander(xs...);
  V = transposed(V);
  auto b = vec<real_t, N>::zeros();
  b(d)   = factorial(d);
  return solve(V, b);
}
//------------------------------------------------------------------------------
/// read her for more information:
/// http://web.media.mit.edu/~crtaylor/calculator.html
template <typename Tensor, floating_point T, size_t N>
auto finite_differences_coefficients(std::size_t                      d,
                                     base_tensor<Tensor, T, N> const& v) {
  auto V           = mat<T, N, N>::vander(v);
  V = transposed(V);
  auto b = vec<T, N>::zeros();
  b(d)   = factorial(d);
  return solve(V, b);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
