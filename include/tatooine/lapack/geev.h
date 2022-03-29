#ifndef TATOOINE_LAPACK_GEEV_H
#define TATOOINE_LAPACK_GEEV_H
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_geev GEEV
/// \brief General Matrix Eigenvalues
/// \ingroup lapack
/// **GEEV** computes for an \f$n\times n\f$ real nonsymmetric matrix \f$\mA\f$, the
/// eigenvalues and, optionally, the left and/or right eigenvectors.
///
/// The right eigenvector v(j) of \f$\mA\f$ satisfies
///
/// \f$\mA\cdot \vv(j) = \lambda(j) \cdot \vv(j)\f$
///
/// where \f$\lambda(j)\f$ is its eigenvalue.
/// The left eigenvector \f$\vu(j)\f$ of \f$\mA\f$ satisfies
///
/// \f$\vu(j)^\dagger \cdot \mA = \lambda(j) \cdot \vu(j)^\dagger\f$
///
/// where \f$\vu(j)^\dagger\f$ denotes the conjugate-transpose of \f$\vu(j)\f$.
///
/// The computed eigenvectors are normalized to have Euclidean norm
/// equal to \f$1\f$ and largest component real.
/// \{
//==============================================================================
template <typename T, size_t N>
auto geev(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W) {
  return ::lapack::geev(Job::NoVec, Job::NoVec, N,
                        A.data(), N, W.data(), nullptr, N, nullptr, N);
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
auto geev_left(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W,
               tensor<T, N, N>& VL) {
  return ::lapack::geev(Job::Vec, Job::NoVec, N,
                        A.data(), N, W.data(), VL.data(), N,
                        nullptr, N);
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
auto geev_right(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W,
                tensor<T, N, N>& VR) {
  return ::lapack::geev(Job::NoVec, Job::Vec, N,
                        A.data(), N, W.data(), nullptr, N, VR.data(),
                        N);
}
//------------------------------------------------------------------------------
template <typename T, size_t N>
auto geev(tensor<T, N, N>& A, tensor<std::complex<T>, N>& W,
          tensor<T, N, N>& VL, tensor<T, N, N>& VR) {
  return ::lapack::geev(Job::Vec, Job::Vec, N, A.data(),
                        N, W.data(), VL.data(), N, VR.data(), N);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
