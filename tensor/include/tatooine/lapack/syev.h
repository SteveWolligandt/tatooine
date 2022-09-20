#ifndef TATOOINE_LAPACK_SYEV_H
#define TATOOINE_LAPACK_SYEV_H
//==============================================================================
extern "C" void dsyev_(char* JOBZ, char* UPLO, int* N, double* A, int* LDA,
                       double* W, double* WORK, int* LWORK, int* INFO);
extern "C" void ssyev_(char* JOBZ, char* UPLO, int* N, float* A, int* LDA,
                       float* W, float* WORK, int* LWORK, int* INFO);
//==============================================================================
#include <tatooine/lapack/base.h>
//==============================================================================
#include <concepts>
#include <memory>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
template <std::floating_point Float>
auto syev(job j, uplo u, int N, Float* A, int LDA, Float* W, Float* WORK,
          int LWORK) -> int {
  auto INFO = int{};
  if constexpr (std::same_as<Float, double>) {
    dsyev_(reinterpret_cast<char*>(&j), reinterpret_cast<char*>(&u), &N, A,
           &LDA, W, WORK, &LWORK, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    ssyev_(reinterpret_cast<char*>(&j), reinterpret_cast<char*>(&u), &N, A,
           &LDA, W, WORK, &LWORK, &INFO);
  }
  return INFO;
}
//------------------------------------------------------------------------------
template <std::floating_point Float>
auto syev(job j, uplo u, int N, Float* A, int LDA, Float* W) -> int {
  auto LWORK = int{-1};
  auto WORK = std::unique_ptr<Float[]>{new Float[1]};
  syev<Float>(j, u, N, A, LDA, W, WORK.get(), LWORK);
  LWORK = static_cast<int>(WORK[0]);
  WORK = std::unique_ptr<Float[]>{new Float[LWORK]};
  return syev<Float>(j, u, N, A, LDA, W, WORK.get(), LWORK);
}
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
auto syev(job jobz, uplo const u, tensor<Real, N, N>& A,
          tensor<Real, N>& W) {
  return syev(jobz, u, N, A.data(), N, W.data());
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
