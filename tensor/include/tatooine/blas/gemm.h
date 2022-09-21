#ifndef TATOOINE_LAPACK_GEMM_H
#define TATOOINE_LAPACK_GEMM_H
//==============================================================================
#include <tatooine/blas/base.h>
#include <concepts>
//==============================================================================
extern "C" {
auto dgemm_(char* transA, char* transB, int* m, int* n, int* k, double* alpha,
            double* A, int* lda, double* B, int* ldb, double* beta, double* C,
            int* ldc) -> void;
auto sgemm_(char* transA, char* transB, int* m, int* n, int* k, float* alpha,
            float* A, int* lda, float* B, int* ldb, float* beta, float* C,
            int* ldc) -> void;
}
//==============================================================================
namespace tatooine::blas {
//==============================================================================
/// \defgroup blas_gemm GEMM
/// \brief General Matrix-Matrix-Multiplication
/// \ingroup blas3
///
///  **GEMM** performs one of the matrix-matrix operations
///
/// \f[\mC=\alpha\cdot\text{op}(\mA)\cdot \text{op}(\mB) + \beta\cdot\mC\f]
///
/// where \f$\text{op}(\mX)\f$ is one of
///
/// \f[\text{op}(\mX)=\mX\ \ \text{ or }\ \ \text{op}(\mX)=\mX^\top\f].
///
/// \f$\alpha\f$ and \f$\beta\f$ are scalars, and \f$\mA\f$, \f$\mB\f$ and
/// \f$\mC\f$ are matrices, with \f$\text{op}(\mA)\f$ an \f$m\times k\f$ matrix,
/// \f$\text{op}(\mB)\f$ a \f$k\times n\f$ matrix and  \f$\mC\f$ an \f$m\times
/// n\f$ matrix.
///
/// - <a
/// href='http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gaeda3cbd99c8fb834a60a6412878226e1.html'>BLAS
/// documentation</a>
/// \{
//==============================================================================
template <std::floating_point Float>
auto gemm(op TRANSA, op TRANSB, int M, int N, int K, Float ALPHA, Float const* A,
          int LDA, Float const* B, int LDB, Float BETA, Float* C, int LDC) -> void {
  if constexpr (std::same_as<Float, double>) {
    dgemm_(reinterpret_cast<char*>(&TRANSA), reinterpret_cast<char*>(&TRANSB),
           &M, &N, &K, &ALPHA, const_cast<Float*>(A), &LDA,
           const_cast<Float*>(B), &LDB, &BETA, C, &LDC);
  } else if constexpr (std::same_as<Float, float>) {
    sgemm_(reinterpret_cast<char*>(&TRANSA), reinterpret_cast<char*>(&TRANSB),
           &M, &N, &K, &ALPHA, const_cast<Float*>(A), &LDA,
           const_cast<Float*>(B), &LDB, &BETA, C, &LDC);
  }
}
//==============================================================================
/// See \ref blas_gemm.
template <std::floating_point Float, std::size_t M, std::size_t N, std::size_t K>
auto gemm(Float const alpha, tensor<Float, M, K> const& A,
          tensor<Float, K, N> const& B, Float const beta, tensor<Float, M, N>& C) {
  return gemm<Float>(op::no_transpose, op::no_transpose, M, N, K, alpha,
                     A.data(), M,
                     B.data(), N, beta, C.data(), M);
}
//------------------------------------------------------------------------------
/// See \ref blas_gemm.
template <std::floating_point Float>
auto gemm(blas::op trans_A, blas::op trans_B, Float const alpha,
          tensor<Float> const& A, tensor<Float> const& B, Float const beta,
          tensor<Float>& C) {
  assert(A.rank() == 2);
  assert(B.rank() == 1 || B.rank() == 2);
  assert(C.rank() == B.rank());
  auto const M = A.dimension(0);
  auto const N = B.rank() == 2 ? B.dimension(1) : 1;
  assert(A.dimension(1) == B.dimension(0));
  auto const K = A.dimension(1);
  assert(C.dimension(0) == M);
  assert(C.rank() == 1 || C.dimension(1) == N);

  return gemm<Float>(trans_A, trans_B, M, N, K, alpha, A.data(), M, B.data(), K,
                     beta, C.data(), M);
}
//------------------------------------------------------------------------------
/// See \ref blas_gemm.
template <std::floating_point Float>
auto gemm(Float const alpha, tensor<Float> const& A, tensor<Float> const& B,
          Float const beta, tensor<Float>& C) {
  return gemm<Float>(op::no_transpose, op::no_transpose, alpha, A, B, beta, C);
}
/// \}
//==============================================================================
}  // namespace tatooine::blas
//==============================================================================
#endif
