#ifndef TATOOINE_TENSOR_OPERATIONS_NORM_H
#define TATOOINE_TENSOR_OPERATIONS_NORM_H
//==============================================================================
#include <tatooine/math.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename T, std::size_t N>
constexpr auto norm_inf(base_tensor<Tensor, T, N> const& t) -> T {
  T norm = -std::numeric_limits<T>::max();
  for (std::size_t i = 0; i < N; ++i) {
    norm = std::max(norm, std::abs(t(i)));
  }
  return norm;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, std::size_t N>
constexpr auto norm1(base_tensor<Tensor, T, N> const& t) {
  return sum(abs(t));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, std::size_t N>
constexpr auto norm(base_tensor<Tensor, T, N> const& t, unsigned p = 2) -> T {
  auto n = T(0);
  for (std::size_t i = 0; i < N; ++i) {
    n += std::pow(t(i), p);
  }
  return std::pow(n, T(1) / T(p));
}
//------------------------------------------------------------------------------
/// squared p-norm of a rank-2 tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto squared_norm(base_tensor<Tensor, T, M, N> const& A,
                            unsigned int const                  p) {
  if (p == 2) {
    return eigenvalues_sym(transposed(A) * A)(N - 1);
  }
  return T(0) / T(0);
}
//------------------------------------------------------------------------------
/// p-norm of a rank-2 tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& A,
                    unsigned int const                  p) {
  return std::sqrt(squared_norm(A, p));
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto squared_norm(base_tensor<Tensor, T, M, N> const& mat,
                            tag::frobenius_t) {
  T n = 0;
  for (std::size_t j = 0; j < N; ++j) {
    for (std::size_t i = 0; i < M; ++i) {
      n += std::abs(mat(i, j));
    }
  }
  return n;
}
//------------------------------------------------------------------------------
/// Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& mat, tag::frobenius_t) {
  return std::sqrt(squared_norm(mat, tag::frobenius));
}
//------------------------------------------------------------------------------
/// 1-norm of a MxN Tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto norm1(base_tensor<Tensor, T, M, N> const& mat) {
  T          max    = -std::numeric_limits<T>::max();
  auto const absmat = abs(mat);
  for (std::size_t i = 0; i < N; ++i) {
    max = std::max(max, sum(absmat.template slice<1>(i)));
  }
  return max;
}
//------------------------------------------------------------------------------
/// infinity-norm of a MxN tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto norm_inf(base_tensor<Tensor, T, M, N> const& mat) {
  T max = -std::numeric_limits<T>::max();
  for (std::size_t i = 0; i < M; ++i) {
    max = std::max(max, sum(abs(mat.template slice<0>(i))));
  }
  return max;
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto squared_norm(base_tensor<Tensor, T, M, N> const& mat) {
  return squared_norm(mat, tag::frobenius);
}
//------------------------------------------------------------------------------
/// squared Frobenius norm of a rank-2 tensor
template <typename Tensor, typename T, std::size_t M, std::size_t N>
constexpr auto norm(base_tensor<Tensor, T, M, N> const& mat) {
  return norm(mat, tag::frobenius);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
