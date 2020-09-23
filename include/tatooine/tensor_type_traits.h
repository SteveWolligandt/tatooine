#ifndef TATOOINE_TENSOR_TYPE_TRAITS_H
#define TATOOINE_TENSOR_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/base_tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor, typename Real, size_t... Dims>
struct num_components<base_tensor<Tensor, Real, Dims...>>
    : std::integral_constant<size_t, (Dims * ...)> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t... Dims>
struct num_components<tensor<Real, Dims...>>
    : std::integral_constant<size_t, (Dims * ...)> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t M, size_t N>
struct num_components<mat<Real, M, N>> : std::integral_constant<size_t, M * N> {
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
struct num_components<vec<Real, N>> : std::integral_constant<size_t, N> {};
//==============================================================================
template <typename Real, size_t N>
struct internal_data_type<vec<Real, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t M, size_t N>
struct internal_data_type<mat<Real, M, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t... Dims>
struct internal_data_type<tensor<Real, Dims...>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t... Dims>
struct internal_data_type<base_tensor<Tensor, Real, Dims...>> {
  using type = Real;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
