#ifndef TATOOINE_LAPACK_GETRF_H
#define TATOOINE_LAPACK_GETRF_H
//==============================================================================
extern "C" {
//==============================================================================
auto dgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO) -> void;
auto sgetrf_(int* M, int* N, double* A, int* LDA, int* IPIV, int* INFO) -> void;
//==============================================================================
}
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_getrf GETRF
/// \brief Computes an LU factorization of a general matrix.
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
//==============================================================================
template <std::floating_point Float>
auto getrf(int M, int N, double* A, int LDA, int* IPIV) -> int {
  auto INFO = int{};
  if constexpr (std::same_as<Float, double>) {
    dgetrf_(&M, &N, A, &LDA, IPIV, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    sgetrf_(&M, &N, A, &LDA, IPIV, &INFO);
  }
  return INFO;
}
//------------------------------------------------------------------------------
template <typename T, size_t M, size_t N>
auto getrf(tensor<T, M, N>& A, tensor<int, tatooine::min(M, N)>& p) {
  return getrf(M, N, A.data(), M, p.data());
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
