#ifndef TATOOINE_LAPACK_ORMQR_H
#define TATOOINE_LAPACK_ORMQR_H
//==============================================================================
extern "C" {
auto dormqr_(char* SIDE, char* TRANS, int* M, int* N, int* K, double* A,
             int* LDA, double* TAU, double* C, int* LDC, double* WORK,
             int* LWORK, int* INFO) -> void;
auto sormqr_(char* SIDE, char* TRANS, int* M, int* N, int* K, float* A,
             int* LDA, float* TAU, float* C, int* LDC, float* WORK, int* LWORK,
             int* INFO) -> void;
}
//==============================================================================
#include <tatooine/lapack/base.h>

#include <concepts>
#include <memory>
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
template <std::floating_point Float>
auto ormqr(side SIDE, op TRANS, int M, int N, int K, Float* A, int LDA,
           Float* TAU, Float* C, int LDC, Float* WORK, int LWORK) -> int {
  auto INFO = int{};
  if constexpr (std::same_as<Float, double>) {
    dormqr_(reinterpret_cast<char*>(&SIDE), reinterpret_cast<char*>(&TRANS), &M,
            &N, &K, A, &LDA, TAU, C, &LDC, WORK, &LWORK, &INFO);
  } else if constexpr (std::same_as<Float, float>) {
    sormqr_(reinterpret_cast<char*>(&SIDE), reinterpret_cast<char*>(&TRANS), &M,
            &N, &K, A, &LDA, TAU, C, &LDC, WORK, &LWORK, &INFO);
  }
  return INFO;
}
//==============================================================================
template <std::floating_point Float>
auto ormqr(side SIDE, op TRANS, int M, int N, int K, Float* A, int LDA,
           Float* TAU, Float* C, int LDC) -> int {
  auto LWORK = int{-1};
  auto WORK  = std::unique_ptr<Float[]>{new Float[1]};

  ormqr<Float>(SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC, WORK.get(), LWORK);
  LWORK = static_cast<int>(WORK[0]);
  WORK  = std::unique_ptr<Float[]>{new Float[LWORK]};
  return ormqr<Float>(SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC, WORK.get(),
                      LWORK);
}
//==============================================================================
/// \defgroup lapack_ormqr ORMQR
/// \brief Orthogonal Matrix
/// \ingroup lapack
///
/// **ORMQR** overwrites the general real \f$m\times n\f$ matrix \f$\mC\f$ with
/// <table>
/// <tr><th>              <th>side = `L` <th>side = `R` </tr>
/// <tr><th>trans = `N`:  <td>\f$\mQ\cdot\mC\f$    <td>\f$\mC\cdot\mQ\f$ </tr>
/// <tr><th>trans = `T`:  <td>\f$\mQ^\top\cdot\mC\f$  <td>\f$\mC\cdot\mQ^\top\f$
/// </tr>
///
/// </table>
/// where \f$\mQ\f$ is a real orthogonal matrix defined as the product of
/// \f$k\f$ elementary reflectors
///
/// \f$\mQ=\mH(1)\cdot\mH(2)\cdot\ldots\cdot\mH(k)\f$
///
/// as returned by \ref lapack_geqrf. \f$\mQ\f$ is of order \f$m\f$ if side =
/// `L` and of order \f$n\f$ if side = `R`
///
/// - <a
/// href='https://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga17b0765a8a0e6547bcf933979b38f0b0.html'>LAPACK
/// documentation</a>
/// \{
//==============================================================================
template <typename T, size_t K, size_t M>
auto ormqr(tensor<T, M, K>& A, tensor<T, M>& c, tensor<T, K>& tau, side const s,
           op trans) {
  return ormqr(s, trans, static_cast<int>(M), 1, static_cast<int>(K), A.data(),
               static_cast<int>(M), tau.data(), c.data(), static_cast<int>(M));
}
//==============================================================================
template <typename T, size_t K, size_t M, size_t N>
auto ormqr(tensor<T, M, K>& A, tensor<T, M, N>& C, tensor<T, K>& tau,
           side const s, op trans) {
  return ormqr(s, trans, static_cast<int>(M), static_cast<int>(N),
               static_cast<int>(K), A.data(), static_cast<int>(M), tau.data(),
               C.data(), static_cast<int>(M));
}
//==============================================================================
template <typename T>
auto ormqr(tensor<T>& A, tensor<T>& C, tensor<T>& tau, side const s, op trans) {
  assert(A.rank() == 2);
  assert(C.rank() == 1 || C.rank() == 2);
  assert(tau.rank() == 1);
  assert(A.dimension(0) == C.dimension(0));
  assert(A.dimension(1) == tau.dimension(0));
  auto const M = A.dimension(0);
  auto const K = A.dimension(1);
  auto const N = C.rank() == 2 ? C.dimension(1) : 1;
  return ormqr(s, trans, static_cast<int>(M), static_cast<int>(N),
               static_cast<int>(K), A.data(), static_cast<int>(M), tau.data(),
               C.data(), static_cast<int>(M));
}
//==============================================================================
/// \}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
