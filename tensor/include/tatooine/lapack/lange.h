#ifndef TATOOINE_LAPACK_LANGE_H
#define TATOOINE_LAPACK_LANGE_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_lange LANGE
/// \brief Calculating norms.
/// \ingroup lapack
/// \{
//==============================================================================
/// Returns the value of the 1-norm, Frobenius norm, infinity-norm, or
/// the largest absolute value of any element of a general rectangular matrix.
/// \param norm Describes which norm will be computed
template <typename T, size_t M, size_t N>
auto lange(tensor<T, M, N>& A, Norm norm) {
  return ::lapack::lange(norm, M, N, A.data(), M);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
