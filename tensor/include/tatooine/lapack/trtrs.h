#ifndef TATOOINE_LAPACK_TRTRS_H
#define TATOOINE_LAPACK_TRTRS_H
//==============================================================================
extern "C" {
auto dtrtrs_(char* UPLO, char* TRANS, char* DIAG, int* N, int* NRHS, double* A,
             int* LDA, double* B, int* LDB, int* INFO) -> void;
//------------------------------------------------------------------------------
auto strtrs_(char* UPLO, char* TRANS, char* DIAG, int* N, int* NRHS, float* A,
             int* LDA, float* B, int* LDB, int* INFO) -> void;
}
//==============================================================================
#include <concepts>
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
template <std::floating_point Float>
auto trtrs(uplo u, op t, diag d, int N, int NRHS, Float* A, int LDA,
           Float* B, int LDB) -> int {
  auto INFO = int{};
  if constexpr (std::same_as<Float, double>) {
    dtrtrs_(reinterpret_cast<char*>(&u), reinterpret_cast<char*>(&t),
            reinterpret_cast<char*>(&d), &N, &NRHS, A, &LDA, B, &LDB, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    strtrs_(reinterpret_cast<char*>(&u), reinterpret_cast<char*>(&t),
            reinterpret_cast<char*>(&d), &N, &NRHS, A, &LDA, B, &LDB, &INFO);
  }
  return INFO;
}
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
auto trtrs(tensor<T, M, N>& A, tensor<T, M, NRHS>& B, uplo const u,
           op const t, diag const d) {
  return trtrs(u, t, d, N, NRHS, A.data(), M, B.data(), M);
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
auto trtrs(tensor<T, M, N>& A, tensor<T, M>& b, uplo const u,
           op const t, diag const d) {
  return trtrs(u, t, d, N, 1, A.data(), M, b.data(), M);
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
auto trtrs(tensor<T>& A, tensor<T>& B, uplo const u,
           op const t,  diag const d) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(A.dimension(0) == B.dimension(0));
  auto const M    = A.dimension(0);
  auto const N    = A.dimension(1);
  auto const NRHS = B.rank() == 2 ? B.dimension(1) : 1;
  return trtrs(u, t, d, N, NRHS, A.data(), M, B.data(), M);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
