#ifndef TATOOINE_BLAS_H
#define TATOOINE_BLAS_H
//==============================================================================
#include <blas.hh>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex Real, size_t... N>
struct tensor;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::blas {
//==============================================================================
template <typename Real, size_t M, size_t N, size_t K>
auto gemm(Real const alpha,
          tensor<Real, M, N> const& A,
          tensor<Real, N, K> const& B,
          Real const beta,
          tensor<Real, M, K>& C) {
  ::blas::gemm(::blas::Layout::ColMajor,
               ::blas::Op::NoTrans,
               ::blas::Op::NoTrans,
               M, K, N,
               alpha,
               A.data_ptr(), M,
               B.data_ptr(), N,
               beta,
               C.data_ptr(), M);
}
//==============================================================================
}  // namespace tatooine::blas
//==============================================================================
#endif
