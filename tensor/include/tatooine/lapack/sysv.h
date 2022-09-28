#ifndef TATOOINE_LAPACK_SYSV_H
#define TATOOINE_LAPACK_SYSV_H
//==============================================================================
extern "C" {
//==============================================================================
auto dsysv_(char* UPLO, int* N, int* NRHS, double* A, int* LDA, int* IPIV,
            double* B, int* LDB, double* WORK, int* LWORK, int* INFO) -> void;
//------------------------------------------------------------------------------
auto ssysv_(char* UPLO, int* N, int* NRHS, float* A, int* LDA, int* IPIV,
            float* B, int* LDB, float* WORK, int* LWORK, int* INFO) -> void;
//==============================================================================
} // extern "C"
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
template <std::floating_point Float>
auto sysv(uplo u, int N, int NRHS, Float* A, int LDA, int* IPIV, Float* B,
          int LDB, Float* WORK, int LWORK) -> int {
  auto INFO = int{};
  if constexpr (std::same_as<Float, double>) {
    dsysv_(reinterpret_cast<char*>(&u), &N, &NRHS, A, &LDA, IPIV, B, &LDB, WORK, &LWORK, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    ssysv_(reinterpret_cast<char*>(&u), &N, &NRHS, A, &LDA, IPIV, B, &LDB, WORK,
           &LWORK, &INFO);
  }
  return INFO;
}
//------------------------------------------------------------------------------
template <std::floating_point Float>
auto sysv(uplo u, int N, int NRHS, Float* A, int LDA, int* IPIV, Float* B,
          int LDB) -> int {
  auto LWORK = int{-1};
  auto WORK  = std::unique_ptr<Float[]>{new Float[1]};
  sysv<Float>(u, N, NRHS, A, LDA, IPIV, B, LDB, WORK.get(), LWORK);
  LWORK = static_cast<int>(WORK[0]);
  WORK  = std::unique_ptr<Float[]>{new Float[LWORK]};
  return sysv<Float>(u, N, NRHS, A, LDA, IPIV, B, LDB, WORK.get(), LWORK);
}
//------------------------------------------------------------------------------
template <std::floating_point Float, size_t N>
auto sysv(tensor<Float, N, N>& A, tensor<Float, N>& b,  uplo const u) {
  auto       ipiv = std::unique_ptr<std::int64_t[]>(new std::int64_t[N]);
  return sysv<Float>(u, N, 1, A.data(), N, ipiv.get(), b.data(),
                        N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::floating_point Float>
auto sysv(tensor<Float>& A, tensor<Float>& B, uplo const u) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(A.dimension(0) == A.dimension(1));
  assert(A.dimension(0) == B.dimension(0));
  auto const N    = A.dimension(0);
  auto       ipiv = std::unique_ptr<int[]>(new int[N]);

  return sysv<Float>(u, static_cast<int>(N),
                     B.rank() == 1 ? 1 : static_cast<int>(B.dimension(1)),
                     A.data(), static_cast<int>(N), ipiv.get(), B.data(),
                     static_cast<int>(N));
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
//template <typename T>
//auto sysv_aa(tensor<T>& A, tensor<T>& B, uplo const u) {
//  assert(A.rank() == 2);
//  assert(B.rank() == 1 || B.rank() == 2);
//  assert(A.dimension(0) == A.dimension(1));
//  assert(A.dimension(0) == B.dimension(0));
//  auto const N    = A.dimension(0);
//  auto       ipiv = std::unique_ptr<std::int64_t[]>(new std::int64_t[N]);
//
//  return sysv_aa(u, N, B.dimension(1), A.data(), N, ipiv.get(),
//                           B.data(), N);
//}
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
//template <typename T>
//auto sysv_rk(tensor<T>& A, tensor<T>& B, uplo const u) {
//  assert(A.rank() == 2);
//  assert(B.rank() == 1 || B.rank() == 2);
//  assert(A.dimension(0) == A.dimension(1));
//  assert(A.dimension(0) == B.dimension(0));
//  auto const N    = A.dimension(0);
//  auto       ipiv = std::unique_ptr<std::int64_t[]>(new std::int64_t[N]);
//
//  return sysv_rk(u, N, B.rank() == 1 ? 1 : B.dimension(1),
//                           A.data(), N, ipiv.get(), B.data(), N);
//}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
