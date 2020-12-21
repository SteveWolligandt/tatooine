#ifndef TATOOINE_TENSOR_LAPACK_UTILITY_H
#define TATOOINE_TENSOR_LAPACK_UTILITY_H
//==============================================================================
#include <tatooine/lapack.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// compute condition number
template <real_or_complex_number T, size_t N, integral P = int>
auto condition_number(const tensor<T, N, N>& A, P const p = 2) {
  if (p == 1) {
    return 1 / lapack::gecon(tensor{A});
  } else if (p == 2) {
    const auto s = singular_values(A);
    return s(0) / s(N-1);
  } else {
    throw std::runtime_error {
      "p = " + std::to_string(p) + " is no valid base. p must be either 1 or 2."
    };
  }
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N, typename PReal>
auto condition_number(const base_tensor<Tensor, T, N, N>& A, PReal p) {
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
template <size_t N>
constexpr auto eigenvalues(tensor<float, N, N> A)
    -> vec<std::complex<float>, N> {
  if constexpr (N == 2) {
    return eigenvalues_22(A);
  } else {
    [[maybe_unused]] lapack_int info;
    std::array<float, N>        wr;
    std::array<float, N>        wi;
    info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                         wr.data(), wi.data(), nullptr, N, nullptr, N);

    vec<std::complex<float>, N> vals;
    for (size_t i = 0; i < N; ++i) {
      vals[i] = {wr[i], wi[i]};
    }
    return vals;
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t N>
constexpr auto eigenvalues(tensor<double, N, N> A)
    -> vec<std::complex<double>, N> {
  if constexpr (N == 2) {
    return eigenvalues_22(A);
  } else {
    [[maybe_unused]] lapack_int info;
    std::array<double, N>       wr;
    std::array<double, N>       wi;
    info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                         wr.data(), wi.data(), nullptr, N, nullptr, N);
    vec<std::complex<double>, N> vals;
    for (size_t i = 0; i < N; ++i) {
      vals[i] = {wr[i], wi[i]};
    }
    return vals;
  }
}
//------------------------------------------------------------------------------
template <size_t N>
auto eigenvectors(tensor<float, N, N> A)
    -> std::pair<mat<std::complex<float>, N, N>, vec<std::complex<float>, N>> {
  [[maybe_unused]] lapack_int info;
  std::array<float, N>        wr;
  std::array<float, N>        wi;
  std::array<float, N * N>    vr;
  info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, vr.data(), N);

  vec<std::complex<float>, N>    vals;
  mat<std::complex<float>, N, N> vecs;
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
//------------------------------------------------------------------------------
template <size_t N>
auto eigenvectors(tensor<double, N, N> A)
    -> std::pair<mat<std::complex<double>, N, N>,
                 vec<std::complex<double>, N>> {
  [[maybe_unused]] lapack_int info;
  std::array<double, N>       wr;
  std::array<double, N>       wi;
  std::array<double, N * N>   vr;
  info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, vr.data(), N);

  vec<std::complex<double>, N>    vals;
  mat<std::complex<double>, N, N> vecs;
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
auto svd(const tensor<T, M, N>& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd(const tensor<T, M, N>& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd(const tensor<T, M, N>& A) {
  return svd(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(const tensor<T, M, N>& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(const tensor<T, M, N>& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_left(const tensor<T, M, N>& A) {
  return svd_left(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(const tensor<T, M, N>& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(const tensor<T, M, N>& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, size_t M, size_t N>
auto svd_right(const tensor<T, M, N>& A) {
  return svd_right(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(const base_tensor<Tensor, T, M, N>& A, tag::full_t /*tag*/) {
  tensor copy{A};
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(const base_tensor<Tensor, T, M, N>& A, tag::economy_t /*tag*/) {
  tensor copy{A};
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(const base_tensor<Tensor, T, M, N>& A) {
  return svd(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(const base_tensor<Tensor, T, M, N>& A, tag::full_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::A, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(const base_tensor<Tensor, T, M, N>& A, tag::economy_t /*tag*/) {
  return lapack::gesvd(tensor{A}, lapack_job::S, lapack_job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(const base_tensor<Tensor, T, M, N>& A) {
  return svd_left(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(const base_tensor<Tensor, T, M, N>& A, tag::full_t /*tag*/) {
  tensor copy{A};
  return gesvd(tensor{A}, lapack_job::N, lapack_job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(const base_tensor<Tensor, T, M, N>& A, tag::economy_t /*tag*/) {
  tensor copy{A};
  return gesvd(tensor{A}, lapack_job::N, lapack_job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(const base_tensor<Tensor, T, M, N>& A) {
  return svd_right(A, tag::full);
}
template <typename Tensor, typename T>
constexpr auto singular_values22(const base_tensor<Tensor, T, 2, 2>& A) {
  const auto a = A(0, 0);
  const auto b = A(0, 1);
  const auto c = A(1, 0);
  const auto d = A(1, 1);

  const auto aa = a * a;
  const auto bb = b * b;
  const auto cc = c * c;
  const auto dd = d * d;
  const auto s1 = aa + bb + cc + dd;
  const auto s2 = std::sqrt((aa + bb - cc - dd) * (aa + bb - cc - dd) +
                            4 * (a * c + b * d) * (a * c + b * d));
  const auto sigma1  = std::sqrt((s1 + s2) / 2);
  const auto sigma2  = std::sqrt((s1 - s2) / 2);
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
constexpr auto singular_values(const tensor<T, M, N>& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return lapack::gesvd(tensor{A}, lapack_job::N, lapack_job::N);
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
constexpr auto singular_values(const base_tensor<Tensor, T, M, N>& A) {
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
