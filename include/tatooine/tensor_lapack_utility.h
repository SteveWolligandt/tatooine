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
template <size_t N>
auto eigenvalues(tensor<float, N, N> A) -> vec<std::complex<float>, N> {
  [[maybe_unused]] lapack_int info;
  std::array<float, N>        wr;
  std::array<float, N>        wi;
  info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, nullptr, N);

  vec<std::complex<float>, N> vals;
  for (size_t i = 0; i < N; ++i) { vals[i] = {wr[i], wi[i]}; }
  return vals;
}
template <size_t N>
auto eigenvalues(tensor<double, N, N> A) -> vec<std::complex<double>, N> {
  [[maybe_unused]] lapack_int info;
  std::array<double, N>       wr;
  std::array<double, N>       wi;
  info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A.data_ptr(), N,
                       wr.data(), wi.data(), nullptr, N, nullptr, N);
  vec<std::complex<double>, N> vals;
  for (size_t i = 0; i < N; ++i) { vals[i] = {wr[i], wi[i]}; }
  return vals;
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
//------------------------------------------------------------------------------
template <size_t N>
auto eigenvalues_sym(tensor<float, N, N> A) {
  vec<float, N>               vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'N', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());
  return vals;
}
template <size_t N>
auto eigenvalues_sym(tensor<double, N, N> A) {
  vec<double, N>              vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());

  return vals;
}

//------------------------------------------------------------------------------
template <size_t N>
auto eigenvectors_sym(mat<float, N, N> A)
    -> std::pair<mat<float, N, N>, vec<float, N>> {
  vec<float, N>               vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());
  return {std::move(A), std::move(vals)};
}
template <size_t N>
auto eigenvectors_sym(mat<double, N, N> A)
    -> std::pair<mat<double, N, N>, vec<double, N>> {
  vec<double, N>              vals;
  [[maybe_unused]] lapack_int info;
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', N, A.data_ptr(), N,
                       vals.data_ptr());
  return {std::move(A), std::move(vals)};
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
//------------------------------------------------------------------------------
template <typename Real, size_t M, size_t N>
auto solve(tensor<Real, M, N> const& A, tensor<Real, N> const& b) {
  return lapack::gesv(A, b);
}
//------------------------------------------------------------------------------
template <typename Real, size_t M, size_t N, size_t O>
auto solve(tensor<Real, M, N> const& A, tensor<Real, N, O> const& B) {
  return lapack::gesv(A, B);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
