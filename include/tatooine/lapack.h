#ifndef TATOOINE_LAPACK_H
#define TATOOINE_LAPACK_H
//==============================================================================
#include <lapack.hh>
#include <tatooine/math.h>
//==============================================================================
namespace tatooine::lapack {
using ::lapack::Uplo;
//==============================================================================
/// \defgroup lapack Lapack
/// 
/// <table>
/// <tr><th>Name	<th>Description</tr>
/// <tr><td>BD <td>bidiagonal matrix</tr>
/// <tr><td>DI <td>diagonal matrix</tr>
/// <tr><td>GB <td>general band matrix</tr>
/// <tr><td>GE <td>general matrix (i.e., unsymmetric, in some cases rectangular)</tr>
/// <tr><td>GG <td>general matrices, generalized problem (i.e., a pair of general matrices)</tr>
/// <tr><td>GT <td>general tridiagonal matrix</tr>
/// <tr><td>HB <td>(complex) Hermitian band matrix</tr>
/// <tr><td>HE <td>(complex) Hermitian matrix</tr>
/// <tr><td>HG <td>upper Hessenberg matrix, generalized problem (i.e. a Hessenberg and a triangular matrix)</tr>
/// <tr><td>HP <td>(complex) Hermitian, packed storage matrix</tr>
/// <tr><td>HS <td>upper Hessenberg matrix</tr>
/// <tr><td>OP <td>(real) orthogonal matrix, packed storage matrix</tr>
/// <tr><td>OR <td>(real) orthogonal matrix</tr>
/// <tr><td>PB <td>symmetric matrix or Hermitian matrix positive definite band</tr>
/// <tr><td>PO <td>symmetric matrix or Hermitian matrix positive definite</tr>
/// <tr><td>PP <td>symmetric matrix or Hermitian matrix positive definite, packed storage matrix</tr>
/// <tr><td>PT <td>symmetric matrix or Hermitian matrix positive definite tridiagonal matrix</tr>
/// <tr><td>SB <td>(real) symmetric band matrix</tr>
/// <tr><td>SP <td>symmetric, packed storage matrix</tr>
/// <tr><td>ST <td>(real) symmetric matrix tridiagonal matrix</tr>
/// <tr><td>SY <td>symmetric matrix</tr>
/// <tr><td>TB <td>triangular band matrix</tr>
/// <tr><td>TG <td>triangular matrices, generalized problem (i.e., a pair of triangular matrices)</tr>
/// <tr><td>TP <td>triangular, packed storage matrix</tr>
/// <tr><td>TR <td>triangular matrix (or in some cases quasi-triangular)</tr>
/// <tr><td>TZ <td>trapezoidal matrix</tr>
/// <tr><td>UN <td>(complex) unitary matrix</tr>
/// <tr><td>UP <td>(complex) unitary, packed storage matrix</tr>
///
/// </table>
/// \{
//==============================================================================
/// \defgroup lapack_getrf GETRF
/// \ingroup lapack
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
/// **GETRF** computes an LU factorization of a general M-by-N matrix A using
/// partial pivoting with row interchanges.
/// \param[in,out] A
///   On entry, the `M-by-N` matrix to be factored.
///   On exit, the factors \f$\mL\f$ and \f$\mU\f$ from the factorization
///   \f$\mA = \mP\cdot\mL\cdot\mU\f$; the unit diagonal elements of \f$\mL\f$
///   are not
/// \param[out] p
///   Diagonal of \f$\mP\f$ that represents the permutation Matrix.
///   The pivot indices; for 1 <= i <= min(M,N), row i of the
///   matrix was interchanged with row p(i).
template <typename T, size_t M, size_t N>
auto getrf(tensor<T, M, N>& A, tensor<int, tatooine::min(M, N)>& p) {
  return ::lapack::getrf(M, N, A.data_ptr(), M, p.data_ptr());
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_gesv GESV General Solve
/// \ingroup lapack
///
/// **DGESV** computes the solution to a real system of linear equations
/// \f$\mA * \mX = \mB\f$
/// where \f$\mA\f$ is an `N-by-N` matrix and \f$\mX\f$ and \f$\mB\f$ are
/// N-by-NRHS matrices.
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
  return ::lapack::gesv(N, 1, A.data_ptr(), N, ipiv.data_ptr(), b.data_ptr(),
                        N);
}
template <typename T, size_t N, size_t K>
auto gesv(tensor<T, N, N>& A,
          tensor<T, N, K>& B,
          tensor<std::int64_t, N>& ipiv) {
  return ::lapack::gesv(N, K, A.data_ptr(), N, ipiv.data_ptr(), B.data_ptr(),
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
                        A.data_ptr(), A.dimension(0), ipiv.data_ptr(),
                        B.data_ptr(), A.dimension(0));
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_sysv SYSV computes the solution to system of linear
/// equations A * X = B for SY matrices.
/// \ingroup lapack
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
/// Computes the solution to a system of linear equations \(A X = B\), where A
/// is an n-by-n symmetric matrix and X and B are n-by-nrhs matrices.
///
/// The diagonal pivoting method is used to factor A as \(A = U D U^T\) if uplo
/// = Upper, or \(A = L D L^T\) if uplo = Lower, where U (or L) is a product of
/// permutation and unit upper (lower) triangular matrices, and D is symmetric
/// and block diagonal with 1-by-1 and 2-by-2 diagonal blocks. The factored form
/// of A is then used to solve the system of equations \(A X = B\).
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
/// \defgroup lapack_geqrf GEQRF General QR Factorization
/// \ingroup lapack
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t M, size_t N>
auto geqrf(tensor<T, M, N>& A, tensor<T, (M < N) ? M : N>& tau) {
  return ::lapack::geqrf(M, N, A.data_ptr(), M, tau.data_ptr());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
auto geqrf(tensor<T>& A, tensor<T>& tau) {
  assert(A.rank() == 2);
  auto const M = A.dimension(0);
  auto const N = A.dimension(1);
  assert(tau.rank() == 1);
  assert(tau.dimension(0) >= tatooine::min(M, N));
  return ::lapack::geqrf(M, N, A.data_ptr(), M, tau.data_ptr());
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_ormqr ORMQR
/// \ingroup lapack
///
/// **ORMQR** overwrites the general real `M x N` matrix `C` with
/// <table>
/// <tr><th>              <th>side = `L` <th>side = `R` </tr>
/// <tr><th>trans = `N`:  <td>`Q * C`    <td>`C * Q`    </tr>
/// <tr><th>trans = `T`:  <td>`Q^T * C`  <td>`C * Q^T`  </tr>
///
/// </table>
/// where `Q` is a real orthogonal matrix defined as the product of `k`
/// elementary reflectors
///
/// `Q = H(1) H(2) . . . H(k)`
///
/// as returned by \ref lapack_geqrf. `Q` is of order `M` if side = `L` and of order `N`
/// if side = `R`
///
/// - <a
/// href='https://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga17b0765a8a0e6547bcf933979b38f0b0.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t K, size_t M>
auto ormqr(tensor<T, M, K>& A, tensor<T, M>& c, tensor<T, K>& tau,
           ::lapack::Side side, ::lapack::Op trans) {
  return ::lapack::ormqr(side, trans, M, 1, K, A.data_ptr(), M, tau.data_ptr(),
                         c.data_ptr(), M);
}
//==============================================================================
template <typename T, size_t K, size_t M, size_t N>
auto ormqr(tensor<T, M, K>& A, tensor<T, M, N>& C, tensor<T, K>& tau,
           ::lapack::Side side, ::lapack::Op trans) {
  return ::lapack::ormqr(side, trans, M, N, K, A.data_ptr(), M, tau.data_ptr(),
                         C.data_ptr(), M);
}
//==============================================================================
template <typename T>
auto ormqr(tensor<T>& A, tensor<T>& C, tensor<T>& tau,
           ::lapack::Side side, ::lapack::Op trans) {
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
/// \defgroup lapack_trtrs TRTRS
/// \ingroup lapack
///
/// **TRTRS** solves a triangular system of the form
///
/// \f$\mA \cdot \mX = \mB\f$
/// or
/// \f$\Transpose{\mA} \cdot \mX = \mB\f$
///
/// where \f$\mathbf{A}\f$ is a triangular matrix of order `N`, and
/// \f$\mathbf{B}\f$ is an `N x RHS` matrix. A check is made to verify that
/// \f$\mathbf{A}\f$ is nonsingular.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga7068947990361e55177155d044435a5c.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
/// \param[in] A The triangular matrix \f$\mA\f$.
/// \param[in,out] B On entry, the right hand side matrix \f$\mB\f$.
///                  On exit, if INFO = 0, the solution matrix \f$\mX\f$.
/// \param uplo A is lower or upper triangular matrix:
/// - `U`: \f$\mA\f$ is upper triangular;
/// - `L`: \f$\mA\f$ is lower triangular.
/// \param diag A is unit (1s on main diagonal) or non-unit
/// - `N`: \f$\mA\f$ is non-unit triangular
/// - `U`: \f$\mA\f$ is unit triangular
template <typename T, size_t M, size_t N, size_t NRHS>
auto trtrs(tensor<T, M, N>& A, tensor<T, M, NRHS>& B, Uplo uplo,
           ::lapack::Op trans, ::lapack::Diag diag) {
  return ::lapack::trtrs(uplo, trans, diag, N, NRHS, A.data_ptr(), M,
                         B.data_ptr(), M);
}
//------------------------------------------------------------------------------
/// \param[in] A The triangular matrix \f$\mA\f$.
/// \param[in,out] b On entry, the right hand side vector \f$\vb\f$.
///                  On exit, the solution vector \f$\vx\f$.
/// \param uplo \f$\mA\f$ is lower or upper triangular matrix:
/// - `U`: \f$\mA\f$ is upper triangular;
/// - `L`: \f$\mA\f$ is lower triangular.
/// \param diag A is unit (1s on main diagonal) or non-unit
/// - `N`: \f$\mA\f$ is non-unit triangular
/// - `U`: \f$\mA\f$ is unit triangular
template <typename T, size_t M, size_t N>
auto trtrs(tensor<T, M, N>& A, tensor<T, M>& b, Uplo uplo,
           ::lapack::Op trans, ::lapack::Diag diag) {
  return ::lapack::trtrs(uplo, trans, diag, N, 1, A.data_ptr(), M, b.data_ptr(),
                         M);
}
//------------------------------------------------------------------------------
/// \param[in] A The triangular matrix \f$\mA\f$.
/// \param[in,out] B On entry, the right hand side matrix \f$\mB\f$.
///                  On exit, if INFO = 0, the solution matrix \f$\mX\f$.
/// \param uplo A is lower or upper triangular matrix:
/// - `U`: \f$\mA\f$ is upper triangular;
/// - `L`: \f$\mA\f$ is lower triangular.
/// \param diag A is unit (1s on main diagonal) or non-unit
/// - `N`: \f$\mA\f$ is non-unit triangular
/// - `U`: \f$\mA\f$ is unit triangular
template <typename T>
auto trtrs(tensor<T>& A, tensor<T>& B, Uplo uplo,
           ::lapack::Op trans, ::lapack::Diag diag) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(A.dimension(0) == B.dimension(0));
  auto const M    = A.dimension(0);
  auto const N    = A.dimension(1);
  auto const NRHS = B.rank() == 2 ? B.dimension(1) : 1;
  return ::lapack::trtrs(uplo, trans, diag, N, NRHS, A.data_ptr(), M,
                         B.data_ptr(), M);
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_lange LANGE
/// \ingroup lapack
/// \{
//==============================================================================
/// Returns the value of the 1-norm, Frobenius norm, infinity-norm, or
/// the largest absolute value of any element of a general rectangular matrix.
/// \param norm Describes which norm will be computed
template <typename T, size_t M, size_t N>
auto lange(tensor<T, M, N>& A, ::lapack::Norm norm) {
  return ::lapack::lange(norm, M, N, A.data_ptr(), M);
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_gecon GECON
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
auto gecon(tensor<T, N, N>& A, ::lapack::Norm norm, T& rcond) {
  auto const n    = lange(A, norm);
  auto       ipiv = tensor<std::int64_t, N>{};
  getrf(A, ipiv);
  return ::lapack::gecon(norm, N, A.data_ptr(), N, n, &rcond);
}
//==============================================================================
/// \}
//==============================================================================
/// \defgroup lapack_syev SYEV
/// \ingroup lapack
/// \{
//==============================================================================
/// Computes all eigenvalues and, optionally, eigenvectors of a real symmetric
/// matrix A.
template <typename Real, size_t N>
auto syev(::lapack::Job jobz, Uplo uplo, tensor<Real, N, N>& A,
          tensor<Real, N>& W) {
  return ::lapack::syev(jobz, uplo, N, A.data_ptr(), N, W.data_ptr());
}
//==============================================================================
template <typename T, size_t N>
auto geev(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W) {
  return ::lapack::geev(::lapack::Job::NoVec, ::lapack::Job::NoVec, N,
                        A.data_ptr(), N, W.data_ptr(), nullptr, N, nullptr, N);
}
template <typename T, size_t N>
auto geev_left(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W,
               tensor<T, N, N>& VL) {
  return ::lapack::geev(::lapack::Job::Vec, ::lapack::Job::NoVec, N,
                        A.data_ptr(), N, W.data_ptr(), VL.data_ptr(), N,
                        nullptr, N);
}
template <typename T, size_t N>
auto geev_right(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W,
                tensor<T, N, N>& VR) {
  return ::lapack::geev(::lapack::Job::NoVec, ::lapack::Job::Vec, N,
                        A.data_ptr(), N, W.data_ptr(), nullptr, N, VR.data_ptr(),
                        N);
}
template <typename T, size_t N>
auto geev(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W,
          tensor<T, N, N>& VL, tensor<T, N, N>& VR) {
  return ::lapack::geev(::lapack::Job::Vec, ::lapack::Job::Vec, N, A.data_ptr(),
                        N, W.data_ptr(), VL.data_ptr(), N, VR.data_ptr(), N);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
