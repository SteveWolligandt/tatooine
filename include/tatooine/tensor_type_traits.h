#ifndef TATOOINE_TENSOR_TYPE_TRAITS_H
#define TATOOINE_TENSOR_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine  {
//==============================================================================
template <typename T>
struct is_tensor : std::false_type {};
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
struct is_tensor<base_tensor<Tensor, Real, Dims...>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t... Dims>
struct is_tensor<tensor<Real, Dims...>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t N>
struct is_tensor<vec<Real, N>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t M, size_t N>
struct is_tensor<mat<Real, M, N>> : std::true_type {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T>
static constexpr auto is_tensor_v = is_tensor<T>::value;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_tensor = std::enable_if_t<(is_tensor_v<Ts> && ...), bool>;
//==============================================================================
template <typename T>
struct is_vector : std::false_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t N>
struct is_vector<vec<Real, N>> : std::true_type {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T>
static constexpr auto is_vector_v = is_vector<T>::value;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_vector = std::enable_if_t<(is_vector_v<Ts> && ...), bool>;
//==============================================================================
template <typename T>
struct is_matrix : std::false_type {};
//------------------------------------------------------------------------------
template <typename Real, size_t M, size_t N>
struct is_matrix<mat<Real, M, N>> : std::true_type {};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename T>
static constexpr auto is_matrix_v = is_matrix<T>::value;
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
using enable_if_matrix = std::enable_if_t<(is_matrix_v<Ts> && ...), bool>;
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t n>
struct num_components<base_tensor<Tensor, Real, n>>
    : std::integral_constant<size_t, n> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t n>
struct num_components<tensor<Real, n>> : std::integral_constant<size_t, n> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t n>
struct num_components<vec<Real, n>> : std::integral_constant<size_t, n> {};

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
