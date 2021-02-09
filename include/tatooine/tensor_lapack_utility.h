#ifndef TATOOINE_TENSOR_LAPACK_UTILITY_H
#define TATOOINE_TENSOR_LAPACK_UTILITY_H
//==============================================================================
#include <tatooine/lapack.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// compute condition number
#ifdef __cpp_concepts
template <typename T, size_t N, integral P = int>
#else
template <typename T, size_t N, typename P = int,
          enable_if<is_integral<P>> = true>
#endif
auto condition_number(tensor<T, N, N> const& A, P const p = 2) {
  if (p == 1) {
    return 1 / lapack::gecon(tensor{A});
  } else if (p == 2) {
    auto const s = singular_values(A);
    return s(0) / s(N-1);
  } else {
    throw std::runtime_error {
      "p = " + std::to_string(p) + " is no valid base. p must be either 1 or 2."
    };
  }
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N, typename PReal>
auto condition_number(base_tensor<Tensor, T, N, N> const& A, PReal p) {
  return condition_number(tensor{A}, p);
}
//==============================================================================
//template <typename Tensor, typename Real>
//constexpr auto eigenvectors_sym(base_tensor<Tensor, Real, 2, 2> const& A) {
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
  return lapack::syev(A);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real>
constexpr auto eigenvalues_22(base_tensor<Tensor, Real, 2, 2> const& A)
    -> vec<std::complex<Real>, 2> {
  decltype(auto) b   = A(1, 0);
  decltype(auto) c   = A(0, 1);
  //if (std::abs(b - c) < 1e-10) {
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
constexpr auto eigenvalues(base_tensor<Tensor, Real, N, N> const& A){
  return eigenvalues(tensor<Real, N, N>{A});
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
constexpr auto eigenvalues(tensor<Real, N, N> A)
    -> vec<std::complex<Real>, N> {
  if constexpr (N == 2) {
    return eigenvalues_22(A);
  } else {
    std::array<Real, N>         wr;
    std::array<Real, N>         wi;
    [[maybe_unused]] auto const info = [&]() {
      if constexpr (std::is_same_v<float, Real>) {
        return LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                             wr.data(), wi.data(), nullptr, N, nullptr, N);
      } else if constexpr (std::is_same_v<double, Real>) {
        return LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                             wr.data(), wi.data(), nullptr, N, nullptr, N);
      }
    }();

    vec<std::complex<Real>, N>
        vals;
    for (size_t i = 0; i < N; ++i) {
      vals[i] = {wr[i], wi[i]};
    }
    return vals;
  }
}
//------------------------------------------------------------------------------
template <typename Tensor,typename Real, size_t N>
auto eigenvectors(base_tensor<Tensor, Real, N, N> const& A){
  return eigenvectors(tensor<Real, N, N>{A});
}
template <typename Real, size_t N>
auto eigenvectors(tensor<Real, N, N> A)
    -> std::pair<mat<std::complex<Real>, N, N>, vec<std::complex<Real>, N>> {
  std::array<Real, N>        wr;
  std::array<Real, N>        wi;
  std::array<Real, N * N>    vr;
  [[maybe_unused]] auto const info = [&]() {
    if constexpr (std::is_same_v<float, Real>) {
      return LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A.data_ptr(), N,
                           wr.data(), wi.data(), nullptr, N, vr.data(), N);
    } else if constexpr (std::is_same_v<double, Real>) {
      return LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A.data_ptr(), N,
                           wr.data(), wi.data(), nullptr, N, vr.data(), N);
    }
  }();

  vec<std::complex<Real>, N>    vals;
  mat<std::complex<Real>, N, N> vecs;
  for (size_t i = 0; i < N; ++i) { vals[i] = {wr[i], wi[i]}; }
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N; ++i) {
      if (wi[j] == 0) {
        vecs(i, j) = {vr[i + j * N], 0};
      } else {
        vecs(i, j)     = {vr[i + j * N], vr[i + (j + 1) * N]};
        vecs(i, j + 1) = {vr[i + j * N], -vr[i + (j + 1) * N]};
        if (i == N - 1) { ++j; }
      }
    }
  }

  return {std::move(vecs), std::move(vals)};
}
//==============================================================================
template <typename T, size_t M, size_t N>
auto svd(tensor<T, M, N> const& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd(tensor<T, M, N> const& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd(tensor<T, M, N> const& A) {
  return svd(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(tensor<T, M, N> const& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(tensor<T, M, N> const& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(tensor<T, M, N> const& A) {
  return svd_left(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(tensor<T, M, N> const& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(tensor<T, M, N> const& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(tensor<T, M, N> const& A) {
  return svd_right(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(base_tensor<Tensor, T, M, N> const& A, tag::full_t /*tag*/) {
  tensor copy{A};
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(base_tensor<Tensor, T, M, N> const& A, tag::economy_t /*tag*/) {
  tensor copy{A};
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(base_tensor<Tensor, T, M, N> const& A) {
  return svd(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(base_tensor<Tensor, T, M, N> const& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(base_tensor<Tensor, T, M, N> const& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(base_tensor<Tensor, T, M, N> const& A) {
  return svd_left(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(base_tensor<Tensor, T, M, N> const& A, tag::full_t /*tag*/) {
  tensor copy{A};
  return gesvd(tensor{A}, lapack_job::N, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(base_tensor<Tensor, T, M, N> const& A, tag::economy_t /*tag*/) {
  tensor copy{A};
  return gesvd(tensor{A}, lapack_job::N, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(base_tensor<Tensor, T, M, N> const& A) {
  return svd_right(A, tag::full);
}
template <typename Tensor, typename T>
constexpr auto singular_values22(base_tensor<Tensor, T, 2, 2> const& A) {
  auto const a = A(0, 0);
  auto const b = A(0, 1);
  auto const c = A(1, 0);
  auto const d = A(1, 1);

  auto const aa = a * a;
  auto const bb = b * b;
  auto const cc = c * c;
  auto const dd = d * d;
  auto const s1 = aa + bb + cc + dd;
  auto const s2 = std::sqrt((aa + bb - cc - dd) * (aa + bb - cc - dd) +
                            4 * (a * c + b * d) * (a * c + b * d));
  auto const sigma1  = std::sqrt((s1 + s2) / 2);
  auto const sigma2  = std::sqrt((s1 - s2) / 2);
  return vec{tatooine::max(sigma1, sigma2),
             tatooine::min(sigma1, sigma2)};
}
//------------------------------------------------------------------------------
template <typename T, size_t M, size_t N>
constexpr auto singular_values(tensor<T, M, N>&& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return gesvd(A, lapack_job::N, lapack_job::N);
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
constexpr auto singular_values(tensor<T, M, N> const& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::N);
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto singular_values(base_tensor<Tensor, T, M, N> const& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return singular_values(tensor{A});
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
