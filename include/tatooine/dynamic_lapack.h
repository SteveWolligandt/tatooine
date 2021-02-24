#ifndef TATOOINE_DYNAMIC_LAPACK_H
#define TATOOINE_DYNAMIC_LAPACK_H
//==============================================================================
#include <tatooine/dynamic_tensor.h>
#include <tatooine/lapack_job.h>
#if TATOOINE_INCLUDE_MKL_LAPACKE
#include <mkl_lapacke.h>
#else
#include <lapacke.h>
#endif
//==============================================================================
namespace tatooine::lapack {
//==============================================================================
#ifdef __cpp_concepts
template <arithmetic T>
#else
template <typename T>
#endif
auto gesv(dynamic_tensor<T> A, dynamic_tensor<T> b) {
  assert(A.num_dimensions() == 2);
  assert(b.num_dimensions() == 1 || b.num_dimensions() == 2);
  assert(A.size(0) == A.size(1));
  assert(A.size(0) == b.size(0));
  if (b.num_dimensions() == 1) {
    auto ipiv = dynamic_tensor<int>::zeros(b.size(0));
    int  nrhs = 1;
    if constexpr (std::is_same_v<double, T>) {
      LAPACKE_dgesv(LAPACK_COL_MAJOR, b.size(0), nrhs, A.data_ptr(), b.size(0),
                    ipiv.data_ptr(), b.data_ptr(), b.size(0));
    } else if constexpr (std::is_same_v<float, T>) {
      LAPACKE_sgesv(LAPACK_COL_MAJOR, b.size(0), nrhs, A.data_ptr(), b.size(0),
                    ipiv.data(), b.data_ptr(), b.size(0));
    } else {
      throw std::runtime_error{"[tatooine::lapack::gesv] - type not accepted"};
    }
  }
  if (b.num_dimensions() == 2) {
    auto ipiv = dynamic_tensor<int>::zeros(b.size(0));
    int  nrhs = b.size(1);
    if constexpr (std::is_same_v<double, T>) {
      LAPACKE_dgesv(LAPACK_COL_MAJOR, b.size(0), nrhs, A.data_ptr(), b.size(0),
                    ipiv.data_ptr(), b.data_ptr(), b.size(0));
    } else if constexpr (std::is_same_v<float, T>) {
      LAPACKE_sgesv(LAPACK_COL_MAJOR, b.size(0), nrhs, A.data_ptr(), b.size(0),
                    ipiv.data(), b.data_ptr(), b.size(0));
    } else {
      throw std::runtime_error{"[tatooine::lapack::gesv] - type not accepted"};
    }
  }
  return b;
}
//==============================================================================
}  // namespace tatooine::lapack
//==============================================================================
#endif
