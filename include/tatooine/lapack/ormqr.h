#ifndef TATOOINE_LAPACK_ORMQR_H
#define TATOOINE_LAPACK_ORMQR_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_ormqr ORMQR
/// \brief Orthogonal Matrix 
/// \ingroup lapack
///
/// **ORMQR** overwrites the general real \f$m\times n\f$ matrix \f$\mC\f$ with
/// <table>
/// <tr><th>              <th>side = `L` <th>side = `R` </tr>
/// <tr><th>trans = `N`:  <td>\f$\mQ\cdot\mC\f$    <td>\f$\mC\cdot\mQ\f$    </tr>
/// <tr><th>trans = `T`:  <td>\f$\mQ^\top\cdot\mC\f$  <td>\f$\mC\cdot\mQ^\top\f$  </tr>
///
/// </table>
/// where \f$\mQ\f$ is a real orthogonal matrix defined as the product of \f$k\f$
/// elementary reflectors
///
/// \f$\mQ=\mH(1)\cdot\mH(2)\cdot\ldots\cdot\mH(k)\f$
///
/// as returned by \ref lapack_geqrf. \f$\mQ\f$ is of order \f$m\f$ if side =
/// `L` and of order \f$n\f$ if side = `R`
///
/// - <a
/// href='https://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga17b0765a8a0e6547bcf933979b38f0b0.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t K, size_t M>
auto ormqr(tensor<T, M, K>& A, tensor<T, M>& c, tensor<T, K>& tau,
           Side const side, Op trans) {
  return ::lapack::ormqr(side, trans, M, 1, K, A.data_ptr(), M, tau.data_ptr(),
                         c.data_ptr(), M);
}
//==============================================================================
template <typename T, size_t K, size_t M, size_t N>
auto ormqr(tensor<T, M, K>& A, tensor<T, M, N>& C, tensor<T, K>& tau,
           Side const side, Op trans) {
  return ::lapack::ormqr(side, trans, M, N, K, A.data_ptr(), M, tau.data_ptr(),
                         C.data_ptr(), M);
}
//==============================================================================
template <typename T>
auto ormqr(tensor<T>& A, tensor<T>& C, tensor<T>& tau,
           Side const side, Op trans) {
  assert(A.rank() == 2);
  assert(C.rank() == 1 || C.rank() == 2);
  assert(tau.rank() == 1);
  assert(A.dimension(0) == C.dimension(0));
  assert(A.dimension(1) == tau.dimension(0));
  auto const M = A.dimension(0);
  auto const K = A.dimension(1);
  auto const N = C.rank() == 2 ? C.dimension(1) : 1;
  return ::lapack::ormqr(side, trans, M, N, K, A.data_ptr(), M, tau.data_ptr(),
                         C.data_ptr(), M);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
