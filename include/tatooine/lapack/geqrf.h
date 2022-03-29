#ifndef TATOOINE_LAPACK_GEQRF_H
#define TATOOINE_LAPACK_GEQRF_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_geqrf GEQRF
/// \brief General QR Factorization
/// \ingroup lapack
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t M, size_t N>
auto geqrf(tensor<T, M, N>& A, tensor<T, (M < N) ? M : N>& tau) {
  return ::lapack::geqrf(M, N, A.data(), M, tau.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
auto geqrf(tensor<T>& A, tensor<T>& tau) {
  assert(A.rank() == 2);
  auto const M = A.dimension(0);
  auto const N = A.dimension(1);
  assert(tau.rank() == 1);
  assert(tau.dimension(0) >= tatooine::min(M, N));
  return ::lapack::geqrf(M, N, A.data(), M, tau.data());
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
