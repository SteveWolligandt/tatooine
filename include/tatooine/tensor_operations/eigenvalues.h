#ifndef TATOOINE_TENSOR_OPERATIONS_EIGENVALUES_H
#define TATOOINE_TENSOR_OPERATIONS_EIGENVALUES_H
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
template <typename Tensor, typename Real, size_t N>
auto eigenvectors_sym(base_tensor<Tensor, Real, N, N> const& A) {
  auto W = std::pair{mat<Real, N, N>{A}, vec<Real, N>{}};
  lapack::syev(::lapack::Job::Vec, ::lapack::Uplo::Upper, W.first, W.second);
  return W;
}
//==============================================================================
template <typename Tensor, typename Real>
constexpr auto eigenvalues_sym(base_tensor<Tensor, Real, 2, 2> const& A) {
  decltype(auto) b = A(1, 0);
  if (std::abs(b) <= 1e-11) {
    return vec<Real, 2>{A(0, 0), A(1, 1)};
  }
  decltype(auto) a      = A(0, 0);
  decltype(auto) d      = A(1, 1);
  auto const     e_sqr  = d * d - 2 * a * d + 4 * b * b + a * a;
  auto const     e      = std::sqrt(e_sqr);
  constexpr auto half   = 1 / Real(2);
  auto           lambda = vec<Real, 2>{-e + d + a, e + d + a} * half;
  if (lambda(0) > lambda(1)) {
    std::swap(lambda(0), lambda(1));
  }
  return lambda;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto eigenvalues_sym(base_tensor<Tensor, Real, N, N> const& A) {
  auto W  = vec<Real, N>{};
  auto A2 = mat<Real, N, N>{A};
  lapack::syev(::lapack::Job::NoVec, ::lapack::Uplo::Upper, A2, W);
  return W;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real>
constexpr auto eigenvalues_22(base_tensor<Tensor, Real, 2, 2> const& A)
    -> vec<std::complex<Real>, 2> {
  decltype(auto) b = A(1, 0);
  decltype(auto) c = A(0, 1);
  // if (std::abs(b - c) < 1e-10) {
  //  return eigenvalues_22_sym(A);
  //}
  decltype(auto) a   = A(0, 0);
  decltype(auto) d   = A(1, 1);
  auto const     sqr = d * d - 2 * a * d + 4 * b * c + a * a;

  vec<std::complex<Real>, 2> s;
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
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t N>
constexpr auto eigenvalues(base_tensor<Tensor, Real, N, N> const& A) {
  return eigenvalues(tensor<Real, N, N>{A});
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
constexpr auto eigenvalues(tensor<Real, N, N> A) -> vec<std::complex<Real>, N> {
  if constexpr (N == 2) {
    return eigenvalues_22(A);
  } else {
    auto                        eig  = tensor<std::complex<Real>, N>{};
    [[maybe_unused]] auto const info = lapack::geev(A, eig);

    return eig;
  }
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
auto eigenvectors(base_tensor<Tensor, Real, N, N> const& A) {
  return eigenvectors(tensor<Real, N, N>{A});
}
template <typename Real, size_t N>
auto eigenvectors(tensor<Real, N, N> A) {
  std::pair<mat<std::complex<Real>, N, N>, vec<std::complex<Real>, N>> eig;
  auto&                       V = eig.first;
  auto&                       W = eig.second;
  mat<Real, N, N>             VR;
  [[maybe_unused]] auto const info = lapack::geev_right(A, W, VR);

  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
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
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
