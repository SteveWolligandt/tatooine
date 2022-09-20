#ifndef TATOOINE_LAPACK_GEQRF_H
#define TATOOINE_LAPACK_GEQRF_H
//==============================================================================
extern "C" {
auto dgeqrf_(int* M, int* N, double* A, int* LDA, double* TAU, double* WORK,
             int* LWORK, int* INFO) -> void;
auto sgeqrf_(int* M, int* N, float* A, int* LDA, float* TAU, float* WORK,
             int* LWORK, int* INFO) -> void;
}
//==============================================================================
#include <tatooine/lapack/base.h>
#include <concepts>
#include <memory>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
template <std::floating_point Float>
auto geqrf(int M, int N, Float* A, int LDA, Float* TAU, Float* WORK, int LWORK)
    -> int {
  auto INFO = int{};
  if constexpr (std::same_as<Float, double>) {
    dgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    sgeqrf_(&M, &N, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }
  return INFO;
}
//==============================================================================
template <std::floating_point Float>
auto geqrf(int M, int N, Float* A, int LDA, Float* TAU) -> int {
  auto LWORK = int{-1};
  auto WORK = std::unique_ptr<Float[]>{new Float[1]};

  geqrf<Float>(M, N, A, LDA, TAU, WORK.get(), LWORK);
  LWORK = static_cast<int>(WORK[0]);
  WORK = std::unique_ptr<Float[]>{new Float[LWORK]};
  return geqrf<Float>(M, N, A, LDA, TAU, WORK.get(), LWORK);
}
//==============================================================================
/// \defgroup lapack_geqrf GEQRF
/// \brief General QR Factorization
/// \ingroup lapack
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t M, size_t N>
auto geqrf(tensor<T, M, N>& A, tensor<T, (M < N) ? M : N>& tau) {
  return geqrf(M, N, A.data(), M, tau.data());
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
auto geqrf(tensor<T>& A, tensor<T>& tau) {
  assert(A.rank() == 2);
  auto const M = A.dimension(0);
  auto const N = A.dimension(1);
  assert(tau.rank() == 1);
  assert(tau.dimension(0) >= tatooine::min(M, N));
  return geqrf(M, N, A.data(), M, tau.data());
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
