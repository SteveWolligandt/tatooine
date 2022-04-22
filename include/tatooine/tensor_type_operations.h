#ifndef TATOOINE_TENSOR_TYPE_OPERATIONS_H
#define TATOOINE_TENSOR_TYPE_OPERATIONS_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/tensor_type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::size_t NewRightDim, typename Tensor>
struct tensor_add_dimension_right_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::size_t NewRightDim, typename Tensor>
using tensor_add_dimension_right =
    typename tensor_add_dimension_right_impl<NewRightDim, Tensor>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::size_t NewRightDim, typename Real, std::size_t... Dims>
struct tensor_add_dimension_right_impl<NewRightDim, tensor<Real, Dims...>> {
  using type = tensor<Real, Dims..., NewRightDim>;
};
template <std::size_t NewRightDim, typename Real, std::size_t M, std::size_t N>
struct tensor_add_dimension_right_impl<NewRightDim, mat<Real, M, N>> {
  using type = tensor<Real, M, N, NewRightDim>;
};
template <std::size_t NewRightDim, typename Real, std::size_t N>
struct tensor_add_dimension_right_impl<NewRightDim, vec<Real, N>> {
  using type = mat<Real, N, NewRightDim>;
};
template <std::size_t NewRightDim>
struct tensor_add_dimension_right_impl<NewRightDim, long double> {
  using type = vec<long double, NewRightDim>;
};
template <std::size_t NewRightDim>
struct tensor_add_dimension_right_impl<NewRightDim, double> {
  using type = vec<double, NewRightDim>;
};
template <std::size_t NewRightDim>
struct tensor_add_dimension_right_impl<NewRightDim, float> {
  using type = vec<float, NewRightDim>;
};
//==============================================================================
template <std::size_t NewLeftDim, typename Tensor>
struct tensor_add_dimension_left_impl;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::size_t NewLeftDim, typename Tensor>
using tensor_add_dimension_left =
    typename tensor_add_dimension_left_impl<NewLeftDim, Tensor>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <std::size_t NewLeftDim, typename Real, std::size_t... Dims>
struct tensor_add_dimension_left_impl<NewLeftDim, tensor<Real, Dims...>> {
  using type = tensor<Real, NewLeftDim, Dims...>;
};
template <std::size_t NewLeftDim, typename Real, std::size_t M, std::size_t N>
struct tensor_add_dimension_left_impl<NewLeftDim, mat<Real, M, N>> {
  using type = tensor<Real, NewLeftDim, M, N>;
};
template <std::size_t NewLeftDim, typename Real, std::size_t N>
struct tensor_add_dimension_left_impl<NewLeftDim, vec<Real, N>> {
  using type = mat<Real, NewLeftDim, N>;
};
template <std::size_t NewLeftDim>
struct tensor_add_dimension_left_impl<NewLeftDim, long double> {
  using type = vec<long double, NewLeftDim>;
};
template <std::size_t NewLeftDim>
struct tensor_add_dimension_left_impl<NewLeftDim, double> {
  using type = vec<double, NewLeftDim>;
};
template <std::size_t NewLeftDim>
struct tensor_add_dimension_left_impl<NewLeftDim, float> {
  using type = vec<float, NewLeftDim>;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
