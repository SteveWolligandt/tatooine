#ifndef TATOOINE_TENSOR_OPERATIONS_H
#define TATOOINE_TENSOR_OPERATIONS_H
//==============================================================================
#include <tatooine/blas.h>
#include <tatooine/lapack.h>
#include <tatooine/mat.h>
#include <tatooine/transposed_tensor.h>
#include <tatooine/vec.h>

#include <optional>

//==============================================================================
#include <tatooine/tensor_operations/binary_operation.h>
#include <tatooine/tensor_operations/cross.h>
#include <tatooine/tensor_operations/determinant.h>
#include <tatooine/tensor_operations/distance.h>
#include <tatooine/tensor_operations/eigenvalues.h>
#include <tatooine/tensor_operations/inv.h>
#include <tatooine/tensor_operations/length.h>
#include <tatooine/tensor_operations/norm.h>
#include <tatooine/tensor_operations/operator_overloads.h>
#include <tatooine/tensor_operations/solve.h>
#include <tatooine/tensor_operations/trace.h>
#include <tatooine/tensor_operations/unary_operation.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// Returns the cosine of the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
constexpr auto cos_angle(base_tensor<Tensor0, T0, N> const& v0,
                         base_tensor<Tensor1, T1, N> const& v1) {
  return dot(normalize(v0), normalize(v1));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, size_t... Dims>
constexpr auto abs(base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [](auto const& component) { return gcem::abs(component); }, t);
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
constexpr auto angle(base_tensor<Tensor0, T0, N> const& v0,
           base_tensor<Tensor1, T1, N> const& v1) {
  return gcem::acos(cos_angle(v0, v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
constexpr auto min_angle(base_tensor<Tensor0, T0, N> const& v0,
               base_tensor<Tensor1, T1, N> const& v1) {
  return gcem::min(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the angle of two normalized vectors.
template <typename Tensor0, typename Tensor1, typename T0, typename T1,
          size_t N>
constexpr auto max_angle(base_tensor<Tensor0, T0, N> const& v0,
               base_tensor<Tensor1, T1, N> const& v1) {
  return gcem::max(angle(v0, v1), angle(v0, -v1));
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, typename T0,
          typename T1, typename T2, size_t N>
constexpr auto cos_angle(base_tensor<Tensor0, T0, N> const& v0,
                         base_tensor<Tensor1, T1, N> const& v1,
                         base_tensor<Tensor2, T2, N> const& v2) {
  return cos_angle(v0 - v1, v2 - v1);
}
//------------------------------------------------------------------------------
/// Returns the cosine of the angle three points.
template <typename Tensor0, typename Tensor1, typename Tensor2, typename T0,
          typename T1, typename T2, size_t N>
constexpr auto angle(base_tensor<Tensor0, T0, N> const& v0,
           base_tensor<Tensor1, T1, N> const& v1,
           base_tensor<Tensor2, T2, N> const& v2) {
  return gcem::acos(cos_angle(v0, v1, v2));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t... Dims>
constexpr auto min(base_tensor<Tensor, T, Dims...> const& t) {
  T m = std::numeric_limits<T>::max();
  t.for_indices([&](auto const... is) { m = gcem::min(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t... Dims>
constexpr auto max(base_tensor<Tensor, T, Dims...> const& t) {
  T m = -std::numeric_limits<T>::max();
  t.for_indices([&](auto const... is) { m = gcem::max(m, t(is...)); });
  return m;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename T, size_t N>
constexpr auto normalize(base_tensor<Tensor, T, N> const& t_in) -> vec<T, N> {
  auto const l = euclidean_length(t_in);
  if (gcem::abs(l) < 1e-13) {
    return vec<T, N>::zeros();
  }
  return t_in / l;
}
//------------------------------------------------------------------------------
/// sum of all components of a vector
template <typename Tensor, typename T, size_t VecDim>
constexpr auto sum(base_tensor<Tensor, T, VecDim> const& v) {
  T s = 0;
  for (size_t i = 0; i < VecDim; ++i) {
    s += v(i);
  }
  return s;
}
//------------------------------------------------------------------------------
template <typename Tensor0, typename T0, typename Tensor1, typename T1,
          size_t N>
constexpr auto dot(base_tensor<Tensor0, T0, N> const& lhs,
                   base_tensor<Tensor1, T1, N> const& rhs) {
  common_type<T0, T1> d = 0;
  for (size_t i = 0; i < N; ++i) {
    d += lhs(i) * rhs(i);
  }
  return d;
}
//------------------------------------------------------------------------------
template <typename T0, typename T1>
constexpr auto reflect(vec<T0, 3> const& incidentVec,
                       vec<T1, 3> const& normal) {
  return incidentVec - 2 * dot(incidentVec, normal) * normal;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename TensorT, size_t... Dims>
constexpr auto sqrt(base_tensor<Tensor, TensorT, Dims...> const& t) {
  return unary_operation(
      [](auto const& component) { return gcem::sqrt(component); }, t);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
