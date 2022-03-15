#ifndef TATOOINE_LAPACK_TRTRS_H
#define TATOOINE_LAPACK_TRTRS_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_trtrs TRTRS
/// \brief Solves triangular systems.
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
auto trtrs(tensor<T, M, N>& A, tensor<T, M, NRHS>& B, Uplo const uplo,
           Op trans, Diag diag) {
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
auto trtrs(tensor<T, M, N>& A, tensor<T, M>& b, Uplo const uplo,
           Op trans, Diag diag) {
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
auto trtrs(tensor<T>& A, tensor<T>& B, Uplo const uplo,
           Op trans, Diag diag) {
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
}  // namespace tatooine::lapack
//==============================================================================
#endif
