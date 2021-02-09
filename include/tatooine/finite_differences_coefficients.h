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
#ifdef __cpp_concepts
template <floating_point... Xs>
#else
template <typename... Xs, enable_if<is_floating_point<Xs...>> = true>
#endif
auto finite_differences_coefficients(std::size_t d, Xs... xs) {
  constexpr auto N = sizeof...(xs);
  using real_t     = common_type<std::decay_t<Xs>...>;
  auto V           = mat<real_t, N, N>::vander(xs...);
  V = transposed(V);
  auto b = vec<real_t, N>::zeros();
  b(d)   = factorial(d);
  return solve(V, b);
}
//------------------------------------------------------------------------------
/// read her for more information:
/// http://web.media.mit.edu/~crtaylor/calculator.html
#ifdef __cpp_concepts
template <typename Tensor, floating_point T, size_t N>
#else
template <typename Tensor, typename T, size_t N,
          enable_if<is_floating_point<T>> = true>
#endif
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
