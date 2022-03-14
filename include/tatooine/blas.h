#ifndef TATOOINE_BLAS_H
#define TATOOINE_BLAS_H
//==============================================================================
#include <blas.hh>
#include <tatooine/concepts.h>
#include <tatooine/transposed_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex Real, size_t... N>
struct tensor;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::blas {
//==============================================================================
/// \defgroup blas BLAS
///
/// <table>
/// <tr><th>Name	<th>Description</tr>
/// <tr><td>BD <td>bidiagonal matrix</tr>
/// <tr><td>DI <td>diagonal matrix</tr>
/// <tr><td>GB <td>general band matrix</tr>
/// <tr><td>GE <td>general matrix (i.e., unsymmetric, in some cases
/// rectangular)</tr> <tr><td>GG <td>general matrices, generalized problem
/// (i.e., a pair of general matrices)</tr> <tr><td>GT <td>general tridiagonal
/// matrix</tr> <tr><td>HB <td>(complex) Hermitian band matrix</tr> <tr><td>HE
/// <td>(complex) Hermitian matrix</tr> <tr><td>HG <td>upper Hessenberg matrix,
/// generalized problem (i.e. a Hessenberg and a triangular matrix)</tr>
/// <tr><td>HP <td>(complex) Hermitian, packed storage matrix</tr>
/// <tr><td>HS <td>upper Hessenberg matrix</tr>
/// <tr><td>OP <td>(real) orthogonal matrix, packed storage matrix</tr>
/// <tr><td>OR <td>(real) orthogonal matrix</tr>
/// <tr><td>PB <td>symmetric matrix or Hermitian matrix positive definite
/// band</tr> <tr><td>PO <td>symmetric matrix or Hermitian matrix positive
/// definite</tr> <tr><td>PP <td>symmetric matrix or Hermitian matrix positive
/// definite, packed storage matrix</tr> <tr><td>PT <td>symmetric matrix or
/// Hermitian matrix positive definite tridiagonal matrix</tr> <tr><td>SB
/// <td>(real) symmetric band matrix</tr> <tr><td>SP <td>symmetric, packed
/// storage matrix</tr> <tr><td>ST <td>(real) symmetric matrix tridiagonal
/// matrix</tr> <tr><td>SY <td>symmetric matrix</tr> <tr><td>TB <td>triangular
/// band matrix</tr> <tr><td>TG <td>triangular matrices, generalized problem
/// (i.e., a pair of triangular matrices)</tr> <tr><td>TP <td>triangular, packed
/// storage matrix</tr> <tr><td>TR <td>triangular matrix (or in some cases
/// quasi-triangular)</tr> <tr><td>TZ <td>trapezoidal matrix</tr> <tr><td>UN
/// <td>(complex) unitary matrix</tr> <tr><td>UP <td>(complex) unitary, packed
/// storage matrix</tr>
///
/// </table>
//==============================================================================
/// \defgroup blas_gemm GEMM
/// \brief General Matrix Matrix Multiplication
/// \ingroup blas
///
///  DGEMM  performs one of the matrix-matrix operations
///
/// \f[C=\alpha\cdot\text{op}(\mA)\cdot \text{op}(\mB) + \beta\cdot\mC\f]
///
/// where \f$\text{op}(\mX)\f$ is one of
///
/// \f[\text{op}(\mX)=\mX\ \ \text{ or }\ \ \text{op}(\mX)=\mX^\top\f].
///
/// \f$\alpha\f$ and \f$\beta\f$ are scalars, and \f$\mA\f$, \f$\mB\f$ and
/// \f$\mC\f$ are matrices, with \f$\text{op}(\mA)\f$ an \f$m\times k\f$ matrix,
/// \f$\text{op}(\mB)\f$ a \f$k\times n\f$ matrix and  \f$\mC\f$ an \f$m\times
/// n\f$ matrix.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html'>BLAS
/// documentation</a>
/// \{
using ::blas::Layout;
using ::blas::Op;
//==============================================================================
/// See \ref blas_gemm.
template <typename Real, size_t M, size_t N, size_t K>
auto gemm(Real const alpha, tensor<Real, M, K> const& A,
          tensor<Real, K, N> const& B, Real const beta, tensor<Real, M, N>& C) {
  return ::blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, M, N, K,
                      alpha, A.data_ptr(), M, B.data_ptr(), N, beta,
                      C.data_ptr(), M);
}
//------------------------------------------------------------------------------
/// See \ref blas_gemm.
template <typename Real>
auto gemm(blas::Op trans_A, blas::Op trans_B, Real const alpha,
          tensor<Real> const& A, tensor<Real> const& B, Real const beta,
          tensor<Real>& C) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(C.rank() == B.rank());
  auto const M = A.dimension(0);
  auto const N = B.rank() == 2 ? B.dimension(1) : 1;
  assert(A.dimension(1) == B.dimension(0));
  auto const K = A.dimension(1);
  assert(C.dimension(0) == M);
  assert(C.rank() == 1 || C.dimension(1) == N);

  return ::blas::gemm(Layout::ColMajor, trans_A, trans_B, M, N, K, alpha,
                      A.data_ptr(), M, B.data_ptr(), K, beta, C.data_ptr(), M);
}
//------------------------------------------------------------------------------
/// See \ref blas_gemm.
template <typename Real>
auto gemm(Real const alpha, tensor<Real> const& A, tensor<Real> const& B,
          Real const beta, tensor<Real>& C) {
  return gemm(Op::NoTrans, Op::NoTrans, alpha, A, B, beta, C);
}
//==============================================================================
}  // namespace tatooine::blas
//==============================================================================
#endif
