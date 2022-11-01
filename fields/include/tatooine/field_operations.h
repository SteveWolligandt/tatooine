#ifndef TATOOINE_FIELD_OPERATIONS_H
#define TATOOINE_FIELD_OPERATIONS_H
//==============================================================================
#include <tatooine/binary_operation_field.h>
#include <tatooine/field.h>
#include <tatooine/unary_operation_field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V0, typename V1, typename Real0, typename Real1,
          std::size_t NumDimensions>
requires(
    V0::num_tensor_components() ==
    V1::num_tensor_components()) constexpr auto dot(vectorfield<V0, Real0,
                                                                NumDimensions> const&
                                                        lhs,
                                                    vectorfield<
                                                        V1, Real1,
                                                        NumDimensions> const&
                                                        rhs) {
  return make_binary_operation_field(
      lhs, rhs, [](auto const& lhs, auto const rhs) { return dot(lhs, rhs); });
}
//------------------------------------------------------------------------------
template <typename V0, typename Real0, typename V1, typename Real1,
          std::size_t N, typename Tensor>
constexpr auto operator+(const field<V0, Real0, N, Tensor>& lhs,
                         const field<V1, Real1, N, Tensor>& rhs) {
  return make_binary_operation_field(
      lhs, rhs, [](auto const& lhs, auto const& rhs) { return lhs + rhs; });
}
//------------------------------------------------------------------------------
template <typename Real0, typename Real1, std::size_t N, std::size_t TM,
          std::size_t TN>
constexpr auto operator*(polymorphic::matrixfield<Real0, N, TM, TN> const& lhs,
                         polymorphic::vectorfield<Real1, N, TN> const& rhs) {
  return make_binary_operation_field(
      lhs, rhs, [](auto const& lhs, auto const& rhs) { return lhs * rhs; });
}
//------------------------------------------------------------------------------
template <typename V0, typename Real0, typename V1, typename Real1,
          std::size_t N, std::size_t TM, std::size_t TN>
constexpr auto operator*(const matrixfield<V0, Real0, N, TM, TN>& lhs,
                         const vectorfield<V1, Real1, N, TN>&     rhs) {
  return make_binary_operation_field(
      lhs, rhs, [](auto const& lhs, auto const& rhs) { return lhs * rhs; });
}
//------------------------------------------------------------------------------
template <typename V0, typename Real0, typename V1, typename Real1,
          std::size_t N, typename Tensor>
constexpr auto operator*(const field<V0, Real0, N, Tensor>& lhs,
                         const field<V1, Real1, N, Tensor>& rhs) {
  return make_binary_operation_field(
      lhs, rhs, [](auto const& lhs, auto const& rhs) { return lhs * rhs; });
}
//------------------------------------------------------------------------------
template <typename V, typename VReal, std::size_t N, typename Tensor>
constexpr auto operator*(field<V, VReal, N, Tensor> const& f,
                         arithmetic auto const             scalar) {
  return V{f.as_derived()} | [scalar](auto const& t) { return t * scalar; };
}
//------------------------------------------------------------------------------
template <typename V, arithmetic VReal, std::size_t N, typename Tensor>
constexpr auto operator*(arithmetic auto const             scalar,
                         field<V, VReal, N, Tensor> const& f) {
  return V{f.as_derived()} | [scalar](auto const& t) { return t * scalar; };
}
//------------------------------------------------------------------------------
template <typename V, typename VReal, std::size_t N,
          arithmetic_or_complex ScalarReal, typename Tensor>
constexpr auto operator/(field<V, VReal, N, Tensor> const& f,
                         ScalarReal const                  scalar) {
  return V{f.as_derived()} | [scalar](auto const& t) { return t / scalar; };
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename V, typename VReal, std::size_t N,
          arithmetic_or_complex ScalarReal, typename Tensor>
constexpr auto operator/(ScalarReal const                  scalar,
                         field<V, VReal, N, Tensor> const& f) {
  return V{f.as_derived()} | [scalar](auto const& t) { return scalar / t; };
}
//------------------------------------------------------------------------------
// template <typename lhs_tensor_type, typename Real0,
//          typename rhs_tensor_type, typename Real1,
//          std::size_t... Dims>
// constexpr auto operator-(
//    const base_tensor<lhs_tensor_type, Real0, Dims...>& lhs,
//    const base_tensor<rhs_tensor_type, Real1, Dims...>& rhs) {
//  return binary_operation(std::minus<common_type<Real0, Real1>>{}, lhs,
//                          rhs);
//}
//
//------------------------------------------------------------------------------
///// matrix-vector-multiplication
// template <typename lhs_tensor_type, typename Real0,
//          typename rhs_tensor_type, typename Real1, std::size_t M, std::size_t
//          N>
// constexpr auto operator*(const base_tensor<lhs_tensor_type, Real0, M, N>&
// lhs,
//                         const base_tensor<rhs_tensor_type, Real1, N>& rhs) {
//  tensor<common_type<Real0, Real1>, M> product;
//  for (std::size_t i = 0; i < M; ++i) {
//    product(i) = dot(lhs.template slice<0>(i), rhs);
//  }
//  return product;
//}
template <typename V, typename VReal, std::size_t N>
constexpr auto squared_euclidean_length(vectorfield<V, VReal, N> const& v) {
  return unary_operation_field{
      v, [](auto const& v) { return squared_euclidean_length(v); }};
}
template <typename V, typename VReal, std::size_t N>
constexpr auto squared_euclidean_length(vectorfield<V, VReal, N>&& v) {
  return unary_operation_field{
      std::move(v), [](auto const& v) { return squared_euclidean_length(v); }};
}
template <typename V, typename VReal, std::size_t N>
constexpr auto euclidean_length(vectorfield<V, VReal, N> const& v) {
  return unary_operation_field{
      v, [](auto const& v) { return euclidean_length(v); }};
}
template <typename V, typename VReal, std::size_t N>
constexpr auto euclidean_length(vectorfield<V, VReal, N>&& v) {
  return unary_operation_field{
      std::move(v), [](auto const& v) { return euclidean_length(v); }};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/Q_field.h>
#include <tatooine/spacetime_vectorfield.h>
#endif
