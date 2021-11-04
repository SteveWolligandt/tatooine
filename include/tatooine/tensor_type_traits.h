#ifndef TATOOINE_TENSOR_TYPE_TRAITS_H
#define TATOOINE_TENSOR_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/internal_value_type.h>
#include <tatooine/num_components.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor>
struct tensor_rank_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto tensor_rank = tensor_rank_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, std::size_t... Dims>
struct tensor_rank_impl<base_tensor<Tensor, Real, Dims...>>
    : std::integral_constant<std::size_t, sizeof...(Dims)> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t... Dims>
struct tensor_rank_impl<tensor<Real, Dims...>>
    : std::integral_constant<std::size_t, sizeof...(Dims)> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t M, std::size_t N>
struct tensor_rank_impl<mat<Real, M, N>>
    : std::integral_constant<std::size_t, 2> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
struct tensor_rank_impl<vec<Real, N>> : std::integral_constant<std::size_t, 1> {
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template<>
struct tensor_rank_impl<double> : std::integral_constant<std::size_t, 0> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template<>
struct tensor_rank_impl<float> : std::integral_constant<std::size_t, 0> {};
//==============================================================================
template <typename Tensor, typename Real, std::size_t... Dims>
struct num_components_impl<base_tensor<Tensor, Real, Dims...>>
    : std::integral_constant<std::size_t, (Dims * ...)> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t... Dims>
struct num_components_impl<tensor<Real, Dims...>>
    : std::integral_constant<std::size_t, (Dims * ...)> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t M, std::size_t N>
struct num_components_impl<mat<Real, M, N>>
    : std::integral_constant<std::size_t, M * N> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t N>
struct num_components_impl<vec<Real, N>>
    : std::integral_constant<std::size_t, N> {};
//==============================================================================
template <typename Real, std::size_t N>
struct internal_value_type_impl<vec<Real, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t M, std::size_t N>
struct internal_value_type_impl<mat<Real, M, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t... Dims>
struct internal_value_type_impl<tensor<Real, Dims...>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, std::size_t... Dims>
struct internal_value_type_impl<base_tensor<Tensor, Real, Dims...>> {
  using type = Real;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
