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
template <std::floating_point... Xs>
auto finite_differences_coefficients(std::size_t d, Xs... xs) {
  constexpr auto N = sizeof...(xs);
  using real_t     = promote_t<std::decay_t<Xs>...>;
  auto V           = mat<real_t, N, N>::vander(xs...);
  std::cerr << V << '\n';
  V = transposed(V);
  std::cerr << V << '\n';
  auto b = vec<real_t, N>::zeros();
  b(d)   = factorial(d);
  std::cerr << b << '\n';
  return solve(V, b);
}
//==============================================================================
}
//==============================================================================
#endif
