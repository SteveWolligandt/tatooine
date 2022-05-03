#ifndef TATOOINE_LAPACK_SYEV_H
#define TATOOINE_LAPACK_SYEV_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_syev SYEV
/// \brief Symmetrical Matrix Eigenvalues
/// \ingroup lapack
/// Computes all eigenvalues and, optionally, eigenvectors of a real symmetric
/// matrix A.
/// \{
//==============================================================================
/// Computes all eigenvalues and, optionally, eigenvectors of a real symmetric
/// matrix A.
template <typename Real, size_t N>
auto syev(Job jobz, Uplo const uplo, tensor<Real, N, N>& A,
          tensor<Real, N>& W) {
  return ::lapack::syev(jobz, uplo, N, A.data(), N, W.data());
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
