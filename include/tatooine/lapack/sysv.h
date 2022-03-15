#ifndef TATOOINE_LAPACK_SYSV_H
#define TATOOINE_LAPACK_SYSV_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_sysv SYSV
/// \brief Computes the solution to symmetric linear systems
/// \ingroup lapack
///
/// Computes the solution to a system of linear equations \f$\mA\mX = \mB\f$,
/// where \f$\mA\f$ is an \f$n\times n\f$ symmetric matrix and \f$\mX\f$ and
/// \f$\mB\f$ are \f$n\times m\f$ matrices.
///
/// The diagonal pivoting method is used to factor \f$\mA\f$ as \f$\mA = \mU \mD
/// \mU^\top\f$ if `uplo = Upper`, or \f$\mA = \mL \mD \mL^\top\f$ if `uplo =
/// Lower`, where \f$\mU\f$ (or \f$\mL\f$) is a product of permutation and unit
/// upper (lower) triangular matrices, and \f$\mD\f$ is symmetric and block
/// diagonal with \f$1\times 1\f$ and \f$2\times 2\f$ diagonal blocks. The
/// factored form of \f$\mA\f$ is then used to solve the system of equations T.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/d6/d0e/group__double_s_ysolve_ga9995c47692c9885ed5d6a6b431686f41.html#ga9995c47692c9885ed5d6a6b431686f41'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t N>
auto sysv(tensor<T, N, N>& A, tensor<T, N>& b,  Uplo const uplo) {
  auto       ipiv = std::unique_ptr<std::int64_t[]>(new std::int64_t[N]);
  return ::lapack::sysv(uplo, N, 1, A.data_ptr(), N, ipiv.get(), b.data_ptr(),
                        N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
auto sysv(tensor<T>& A, tensor<T>& B, Uplo const uplo) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(A.dimension(0) == A.dimension(1));
  assert(A.dimension(0) == B.dimension(0));
  auto const N    = A.dimension(0);
  auto       ipiv = std::unique_ptr<std::int64_t[]>(new std::int64_t[N]);

  return ::lapack::sysv(uplo, N, B.rank() == 1 ? 1 : B.dimension(1),
                        A.data_ptr(), N, ipiv.get(), B.data_ptr(), N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// Computes the solution to a system of linear equations \(A X = B\), where A
/// is an n-by-n symmetric matrix and X and B are n-by-nrhs matrices.
///
/// Aasen's algorithm is used to factor A as \(A = U T U^T\) if uplo = Upper, or
/// \(A = L T L^T\) if uplo = Lower, where U (or L) is a product of permutation
/// and unit upper (lower) triangular matrices, and T is symmetric tridiagonal.
/// The factored form of A is then used to solve the system of equations \(A X =
/// B\).
template <typename T>
auto sysv_aa(tensor<T>& A, tensor<T>& B, Uplo const uplo) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(A.dimension(0) == A.dimension(1));
  assert(A.dimension(0) == B.dimension(0));
  auto const N    = A.dimension(0);
  auto       ipiv = std::unique_ptr<std::int64_t[]>(new std::int64_t[N]);

  return ::lapack::sysv_aa(uplo, N, B.dimension(1), A.data_ptr(), N, ipiv.get(),
                           B.data_ptr(), N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
/// Computes the solution to a system of linear equations.
///
///\[ A X = B, \]
///
/// where A is an n-by-n symmetric matrix and X and B are n-by-nrhs matrices.
///
/// The bounded Bunch-Kaufman (rook) diagonal pivoting method is used to factor
/// A as \(A = P U D U^T P^T\) if uplo = Upper, or \(A = P L D L^T P^T\) if uplo
/// = Lower, where U (or L) is unit upper (or lower) triangular matrix, \(U^T\)
/// (or
///    \(L^T\)) is the transpose of U (or L), P is a permutation matrix, \(P^T\)
/// is the transpose of P, and D is symmetric and block diagonal with 1-by-1 and
/// 2-by-2 diagonal blocks.
///
/// lapack::sytrf_rk is called to compute the factorization of a symmetric
/// matrix. The factored form of A is then used to solve the system of equations
/// \(A X = B\) by calling lapack::sytrs_rk.
template <typename T>
auto sysv_rk(tensor<T>& A, tensor<T>& B, Uplo const uplo) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(A.dimension(0) == A.dimension(1));
  assert(A.dimension(0) == B.dimension(0));
  auto const N    = A.dimension(0);
  auto       ipiv = std::unique_ptr<std::int64_t[]>(new std::int64_t[N]);

  return ::lapack::sysv_rk(uplo, N, B.rank() == 1 ? 1 : B.dimension(1),
                           A.data_ptr(), N, ipiv.get(), B.data_ptr(), N);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
