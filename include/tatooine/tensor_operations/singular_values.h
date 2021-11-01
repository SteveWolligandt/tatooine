#ifndef TATOOINE_TENSOR_OPERATIONS_SINGULAR_VALUES_H
#define TATOOINE_TENSOR_OPERATIONS_SINGULAR_VALUES_H
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(base_tensor<Tensor, T, M, N> const& A_base, tag::full_t[ > tag < ]) {
  auto A = mat<T, M, N>{A_base};
  return lapack::gesvd(A, lapack::job::A, lapack::job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(base_tensor<Tensor, T, M, N> const& A_base,
         tag::economy_t[ > tag < ]) {
  auto A = mat<T, M, N>{A_base};
  return lapack::gesvd(A, lapack::job::S, lapack::job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd(base_tensor<Tensor, T, M, N> const& A) {
  return svd(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(base_tensor<Tensor, T, M, N> const& A_base,
              tag::full_t[ > tag < ]) {
  auto A = mat<T, M, N>{A_base};
  return lapack::gesvd(A, lapack::job::A, lapack::job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(base_tensor<Tensor, T, M, N> const& A_base,
              tag::economy_t[ > tag < ]) {
  auto A = mat<T, M, N>{A_base};
  return lapack::gesvd(A, lapack::job::S, lapack::job::N);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_left(base_tensor<Tensor, T, M, N> const& A) {
  return svd_left(A, tag::full);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(base_tensor<Tensor, T, M, N> const& A_base,
               tag::full_t[ > tag < ]) {
  auto A = mat<T, M, N>{A_base};
  return lapack::gesvd(A, lapack::job::N, lapack::job::A);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(base_tensor<Tensor, T, M, N> const& A_base,
               tag::economy_t[ > tag < ]) {
  auto A = mat<T, M, N>{A_base};
  return lapack::gesvd(A, lapack::job::N, lapack::job::S);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
auto svd_right(base_tensor<Tensor, T, M, N> const& A) {
  return svd_right(A, tag::full);
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T>
constexpr auto singular_values22(base_tensor<Tensor, T, 2, 2> const& A) {
  auto const a = A(0, 0);
  auto const b = A(0, 1);
  auto const c = A(1, 0);
  auto const d = A(1, 1);

  auto const aa     = a * a;
  auto const bb     = b * b;
  auto const cc     = c * c;
  auto const dd     = d * d;
  auto const s1     = aa + bb + cc + dd;
  auto const s2     = std::sqrt((aa + bb - cc - dd) * (aa + bb - cc - dd) +
                            4 * (a * c + b * d) * (a * c + b * d));
  auto const sigma1 = std::sqrt((s1 + s2) / 2);
  auto const sigma2 = std::sqrt((s1 - s2) / 2);
  return vec{tatooine::max(sigma1, sigma2), tatooine::min(sigma1, sigma2)};
}
//------------------------------------------------------------------------------
template <typename T, size_t M, size_t N>
constexpr auto singular_values(tensor<T, M, N>&& A) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A);
  } else {
    return gesvd(A, lapack::job::N, lapack::job::N);
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename T, size_t M, size_t N>
    constexpr auto singular_values(base_tensor<Tensor, T, M, N> const& A_base) {
  if constexpr (M == 2 && N == 2) {
    return singular_values22(A_base);
  } else {
    auto A = mat<T, M, N>{A_base};
    return lapack::gesvd(A, lapack::job::N, lapack::job::N);
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
