#ifndef TATOOINE_LAPACK_GEEV_H
#define TATOOINE_LAPACK_GEEV_H
//==============================================================================
extern "C" {
auto dgeev_(char* JOBVL, char* JOBVR, int* N, double* A, int* LDA, double* WR,
            double* WI, double* VL, int* LDVL, double* VR, int* LDVR,
            double* WORK, int* LWORK, int* INFO) -> void;

auto sgeev_(char* JOBVL, char* JOBVR, int* N, float* A, int* LDA, float* WR,
            float* WI, float* VL, int* LDVL, float* VR, int* LDVR, float* WORK,
            int* LWORK, int* INFO) -> void;
}
//==============================================================================
#include <tatooine/lapack/base.h>

#include <concepts>
#include <complex>
#include <memory>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
/// \defgroup lapack_geev GEEV
/// \brief General Matrix Eigenvalues
/// \ingroup lapack
/// **GEEV** computes for an \f$n\times n\f$ real nonsymmetric matrix \f$\mA\f$,
/// the eigenvalues and, optionally, the left and/or right eigenvectors.
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
template <std::floating_point Float>
auto geev(job JOBVL, job JOBVR, int N, Float* A, int LDA, Float* WR,
          Float* WI, Float* VL, int LDVL, Float* VR, int LDVR, Float* WORK,
          int LWORK) -> int {
  auto INFO = int{};
  if constexpr (std::same_as<Float, double>) {
    dgeev_(reinterpret_cast<char*>(&JOBVL), reinterpret_cast<char*>(&JOBVR), &N,
           A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    sgeev_(reinterpret_cast<char*>(&JOBVL), reinterpret_cast<char*>(&JOBVR), &N,
           A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
  }
  return INFO;
}
//------------------------------------------------------------------------------
template <std::floating_point Float>
auto geev(job const JOBVL, job const JOBVR, int N, Float* A, int LDA, Float* WR,
          Float* WI, Float* VL, int LDVL, Float* VR, int LDVR) -> int {
  auto LWORK = int{-1};
  auto WORK  = std::unique_ptr<Float[]>{new Float[1]};
  geev<Float>(JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR, LDVR, WORK.get(),
              LWORK);
  LWORK = static_cast<int>(WORK[0]);
  WORK  = std::unique_ptr<Float[]>{new Float[LWORK]};
  return geev<Float>(JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR, LDVR,
                     WORK.get(), LWORK);
}
//------------------------------------------------------------------------------
template <std::floating_point Float>
auto geev(job JOBVL, job JOBVR, int N, Float* A, int LDA,
          std::complex<Float>* W, Float* VL, int LDVL, Float* VR, int LDVR,
          Float* WORK, int LWORK) -> int {
  auto INFO = int{};
  auto WRI = std::unique_ptr<Float[]>{new Float[N*2]};
  if constexpr (std::same_as<Float, double>) {
    dgeev_(reinterpret_cast<char*>(&JOBVL), reinterpret_cast<char*>(&JOBVR), &N,
           A, &LDA, WRI.get(), WRI.get() + N, VL, &LDVL, VR, &LDVR, WORK,
           &LWORK, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    sgeev_(reinterpret_cast<char*>(&JOBVL), reinterpret_cast<char*>(&JOBVR), &N,
           A, &LDA, WRI.get(), WRI.get() + N, VL, &LDVL, VR, &LDVR, WORK,
           &LWORK, &INFO);
  }
  for (int i = 0; i < N; ++i) {
    W[i].real(WRI[i]);
    W[i].imag(WRI[i+N]);
  }
  return INFO;
}
//------------------------------------------------------------------------------
template <std::floating_point Float>
auto geev(job const JOBVL, job const JOBVR, int N, Float* A, int LDA,
          std::complex<Float>* W, Float* VL, int LDVL, Float* VR, int LDVR)
    -> int {
  auto LWORK = int{-1};
  auto WORK  = std::unique_ptr<Float[]>{new Float[1]};
  geev<Float>(JOBVL, JOBVR, N, A, LDA, W, VL, LDVL, VR, LDVR, WORK.get(),
              LWORK);
  LWORK = static_cast<int>(WORK[0]);
  WORK  = std::unique_ptr<Float[]>{new Float[LWORK]};
  return geev<Float>(JOBVL, JOBVR, N, A, LDA, W, VL, LDVL, VR, LDVR, WORK.get(),
                     LWORK);
}
//------------------------------------------------------------------------------
template <std::floating_point Float, size_t N>
auto geev(tensor<Float, N, N>& A, tensor<std::complex<Float>, N>& W) {
  return geev<Float>(job::no_vec, job::no_vec, N, A.data(), N, W.data(), nullptr, N,
              nullptr, N);
}
//------------------------------------------------------------------------------
template <std::floating_point Float, size_t N>
auto geev_left(tensor<Float, N, N>& A, tensor<std::complex<Float>, N>& W,
               tensor<Float, N, N>& VL) {
  return geev<Float>(job::vec, job::no_vec, N, A.data(), N, W.data(), VL.data(), N,
              nullptr, N);
}
//------------------------------------------------------------------------------
template <std::floating_point Float, size_t N>
auto geev_right(tensor<Float, N, N>& A, tensor<std::complex<Float>, N>& W,
                tensor<Float, N, N>& VR) {
  return geev<Float>(job::no_vec, job::vec, N, A.data(), N, W.data(), nullptr, N,
              VR.data(), N);
}
//------------------------------------------------------------------------------
template <std::floating_point Float, size_t N>
auto geev(tensor<Float, N, N>& A, tensor<std::complex<Float>, N>& W,
          tensor<Float, N, N>& VL, tensor<Float, N, N>& VR) {
  return geev<Float>(job::vec, job::vec, N, A.data(), N, W.data(), VL.data(), N,
              VR.data(), N);
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
