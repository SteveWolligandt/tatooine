#ifndef TATOOINE_TENSOR_OPERATIONS_EIGENVALUES_H
#define TATOOINE_TENSOR_OPERATIONS_EIGENVALUES_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/tensor_typedefs.h>
#include <tatooine/vec_typedefs.h>
#include <tatooine/mat_typedefs.h>
//==============================================================================
namespace tatooine {
//==============================================================================
// template <typename Tensor, typename Real>
// constexpr auto eigenvectors_sym(base_tensor<Tensor, Real, 2, 2> const& A) {
//  decltype(auto) b     = A(1, 0);
//  if (b == 0) {
//    return std::pair{mat<Real, 2, 2>::eye(), vec<Real, 2>{A(0, 0), A(1, 1)}};
//  }
//
//  decltype(auto) a     = A(0, 0);
//  decltype(auto) d     = A(1, 1);
//  auto const     e_sqr = d * d - 2 * a * d + 4 * b * b + a * a;
//  auto const     e     = std::sqrt(e_sqr);
//  constexpr auto half  = 1 / Real(2);
//  auto const     b2inv = 1 / (2 * b);
//  std::pair      out{mat<Real, 2, 2>{{Real(1), Real(1)},
//                                {-(e - d + a) * b2inv, (e + d - a) * b2inv}},
//                vec<Real, 2>{-(e - d - a) * half, (e + d + a) * half}};
//  if (out.second(0) > out.second(1)) {
//    std::swap(out.first(1, 0), out.first(1, 1));
//    std::swap(out.second(0), out.second(1));
//  }
//  if (out.first(1, 0) < 0) {
//    out.first.col(0) *= -1;
//  }
//  if (out.first(1, 1) < 0) {
//    out.first.col(1) *= -1;
//  }
//  out.first.col(0) /= std::sqrt(1 + out.first(1, 0) * out.first(1, 0));
//  out.first.col(1) /= std::sqrt(1 + out.first(1, 1) * out.first(1, 1));
//  return out;
//}
//------------------------------------------------------------------------------
#if TATOOINE_BLAS_AND_LAPACK_AVAILABLE
template <static_quadratic_mat Mat>
auto eigenvectors_sym(Mat&& A) {
  static constexpr auto N = tensor_dimension<Mat, 0>;

  auto W = std::pair{mat<tensor_value_type<Mat>, N, N>{std::forward<Mat>(A)},
                     vec<tensor_value_type<Mat>, N>{}};
  lapack::syev(lapack::job::vec, lapack::uplo::upper, W.first, W.second);
  return W;
}
#endif
//==============================================================================
template <fixed_size_quadratic_mat<2> Mat>
constexpr auto eigenvalues_sym(Mat&& A) {
  decltype(auto) b = A(1, 0);
  if (std::abs(b) <= 1e-11) {
    return vec<tensor_value_type<Mat>, 2>{A(0, 0), A(1, 1)};
  }
  decltype(auto) a     = A(0, 0);
  decltype(auto) d     = A(1, 1);
  auto const     e_sqr = d * d - 2 * a * d + 4 * b * b + a * a;
  auto const     e     = std::sqrt(e_sqr);
  constexpr auto half  = 1 / tensor_value_type<Mat>(2);
  auto lambda = vec<tensor_value_type<Mat>, 2>{-e + d + a, e + d + a} * half;
  if (lambda(0) > lambda(1)) {
    std::swap(lambda(0), lambda(1));
  }
  return lambda;
}
//------------------------------------------------------------------------------
#if TATOOINE_BLAS_AND_LAPACK_AVAILABLE
template <static_quadratic_mat Mat>
constexpr auto eigenvalues_sym(Mat&& A) {
  auto constexpr N = tensor_dimensions<Mat>[0];
  auto W           = vec<tensor_value_type<Mat>, N>{};
  auto A2          = mat<tensor_value_type<Mat>, N, N>{std::forward<Mat>(A)};
  lapack::syev(lapack::job::no_vec, lapack::uplo::upper, A2, W);
  return W;
}
#endif
//------------------------------------------------------------------------------
#if TATOOINE_BLAS_AND_LAPACK_AVAILABLE
template <static_quadratic_mat Mat>
constexpr auto eigenvalues(Mat&& A) {
  using Real       = tensor_value_type<Mat>;
  auto constexpr N = tensor_dimensions<Mat>[0];
  auto A2  = tensor<Real, N, N>{std::forward<Mat>(A)};
  auto eig = complex_tensor<Real, N>{};
  [[maybe_unused]] auto const info = lapack::geev(A2, eig);
  return eig;
}
#endif
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <fixed_size_quadratic_mat<2> Mat>
constexpr auto eigenvalues(Mat&& A)
    -> complex_vec<tensor_value_type<Mat>, 2> {
  using value_t    = tensor_value_type<Mat>;
  decltype(auto) b = A(1, 0);
  decltype(auto) c = A(0, 1);
  // if (std::abs(b - c) < 1e-10) {
  //  return eigenvalues_22_sym(A);
  //}
  decltype(auto) a   = A(0, 0);
  decltype(auto) d   = A(1, 1);
  auto const     sqr = d * d - 2 * a * d + 4 * b * c + a * a;

  auto s = ComplexVec2<value_t>{};
  if (sqr >= 0) {
    s(0).real(-(std::sqrt(sqr) - d - a) / 2);
    s(1).real((std::sqrt(sqr) + d + a) / 2);
  } else {
    s(0).real((d + a) / 2);
    s(1).real(s(0).real());
    s(0).imag(std::sqrt(std::abs(sqr)) / 2);
    s(1).imag(-s(0).imag());
  }
  return s;
}
//------------------------------------------------------------------------------
#if TATOOINE_BLAS_AND_LAPACK_AVAILABLE
template <static_quadratic_mat Mat>
auto eigenvectors(Mat&& B) {
  auto constexpr N = tensor_dimensions<Mat>[0];
  using Real       = tensor_value_type<Mat>;
  auto  A          = tensor<Real, N, N>{B};
  auto  eig = std::pair{complex_mat<Real, N, N>{}, complex_vec<Real, N>{}};
  auto& V   = eig.first;
  auto& W   = eig.second;
  auto  VR  = mat<Real, N, N>{};
  [[maybe_unused]] auto const info = lapack::geev_right(A, W, VR);

  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < N; ++i) {
      if (W[j].imag() == 0) {
        V(i, j) = {VR[i + j * N], 0};
      } else {
        V(i, j)     = {VR[i + j * N], VR[i + (j + 1) * N]};
        V(i, j + 1) = {VR[i + j * N], -VR[i + (j + 1) * N]};
        if (i == N - 1) {
          ++j;
        }
      }
    }
  }

  return eig;
}
#endif
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
