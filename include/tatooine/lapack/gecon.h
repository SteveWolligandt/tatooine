#ifndef TATOOINE_LAPACK_GECON_H
#define TATOOINE_LAPACK_GECON_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_gecon GECON
/// \brief Estimates the reciprocal of the condition number.
/// \ingroup lapack
///
/// **GECON** estimates the reciprocal of the condition number of a general
/// real matrix A, in either the 1-norm or the infinity-norm, using
/// the LU factorization computed by DGETRF.
///
/// An estimate is obtained for norm(inv(A)), and the reciprocal of the
/// condition number is computed as
///    RCOND = 1 / ( norm(A) * norm(inv(A)) ).
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga188b8d30443d14b1a3f7f8331d87ae60.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t N>
auto gecon(tensor<T, N, N>& A, Norm norm, T& rcond) {
  auto const n    = lange(A, norm);
  auto       ipiv = tensor<std::int64_t, N>{};
  getrf(A, ipiv);
  return ::lapack::gecon(norm, N, A.data_ptr(), N, n, &rcond);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
