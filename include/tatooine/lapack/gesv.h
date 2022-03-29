#ifndef TATOOINE_LAPACK_GESV_H
#define TATOOINE_LAPACK_GESV_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_gesv GESV
/// \brief General Solve
/// \ingroup lapack
///
/// **GESV** computes the solution to a real system of linear equations
/// \f$\mA\mX = \mB\f$
/// where \f$\mA\f$ is an \f$n\times n\f$ matrix and \f$\mX\f$ and \f$\mB\f$ are
/// \f$n\times m\f$ matrices.
///
/// The LU decomposition with partial pivoting and row interchanges is
/// used to factor \f$\mA\f$ as
/// \f$\mA  = \mP\cdot\mL\cdot\mU\f$
/// where \f$\mP\f$ is a permutation matrix, \f$\mL\f$ is unit lower triangular,
/// and \f$\mU\f$ is upper triangular.  The factored form of \f$\mA\f$ is then
/// used to solve the system of equations \f$\mA\cdot\mX=\mB\f$.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_ga5ee879032a8365897c3ba91e3dc8d512.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t N>
auto gesv(tensor<T, N, N>& A,
          tensor<T, N>& b,
          tensor<std::int64_t, N>& ipiv) {
  return ::lapack::gesv(N, 1, A.data(), N, ipiv.data(), b.data(),
                        N);
}
template <typename T, size_t N, size_t K>
auto gesv(tensor<T, N, N>& A,
          tensor<T, N, K>& B,
          tensor<std::int64_t, N>& ipiv) {
  return ::lapack::gesv(N, K, A.data(), N, ipiv.data(), B.data(),
                        N);
}
template <typename T>
auto gesv(tensor<T>& A, tensor<T>& B, tensor<std::int64_t>& ipiv) {
  assert(A.rank() == 2);
  assert(A.dimension(0) == A.dimension(1));

  assert(B.rank() > 0);
  assert(B.rank() <= 2);

  assert(A.dimension(0) == B.dimension(0));

  ipiv.resize(A.dimension(0));
  return ::lapack::gesv(A.dimension(0), (B.rank() == 1 ? 1 : B.dimension(1)),
                        A.data(), A.dimension(0), ipiv.data(),
                        B.data(), A.dimension(0));
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
