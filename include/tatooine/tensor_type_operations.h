#ifndef TATOOINE_TENSOR_TYPE_OPERATIONS_H
#define TATOOINE_TENSOR_TYPE_OPERATIONS_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/internal_value_type.h>
#include <tatooine/num_components.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t NewRightDim, typename Tensor>
struct tensor_add_dimension_right;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t NewRightDim, typename Tensor>
using tensor_add_dimension_right_t =
    typename tensor_add_dimension_right<NewRightDim, Tensor>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t NewRightDim, typename Real, size_t... Dims>
struct tensor_add_dimension_right<NewRightDim, tensor<Real, Dims...>> {
  using type = tensor<Real, Dims..., NewRightDim>;
};
template <size_t NewRightDim, typename Real, size_t M, size_t N>
struct tensor_add_dimension_right<NewRightDim, mat<Real, M, N>> {
  using type = tensor<Real, M, N, NewRightDim>;
};
template <size_t NewRightDim, typename Real, size_t N>
struct tensor_add_dimension_right<NewRightDim, vec<Real, N>> {
  using type = mat<Real, N, NewRightDim>;
};
template <size_t NewRightDim>
struct tensor_add_dimension_right<NewRightDim, long double> {
  using type = vec<long double, NewRightDim>;
};
template <size_t NewRightDim>
struct tensor_add_dimension_right<NewRightDim, double> {
  using type = vec<double, NewRightDim>;
};
template <size_t NewRightDim>
struct tensor_add_dimension_right<NewRightDim, float> {
  using type = vec<float, NewRightDim>;
};
//==============================================================================
template <size_t NewLeftDim, typename Tensor>
struct tensor_add_dimension_left;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t NewLeftDim, typename Tensor>
using tensor_add_dimension_left_t =
    typename tensor_add_dimension_left<NewLeftDim, Tensor>::type;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <size_t NewLeftDim, typename Real, size_t... Dims>
struct tensor_add_dimension_left<NewLeftDim, tensor<Real, Dims...>> {
  using type = tensor<Real, NewLeftDim, Dims...>;
};
template <size_t NewLeftDim, typename Real, size_t M, size_t N>
struct tensor_add_dimension_left<NewLeftDim, mat<Real, M, N>> {
  using type = tensor<Real, NewLeftDim, M, N>;
};
template <size_t NewLeftDim, typename Real, size_t N>
struct tensor_add_dimension_left<NewLeftDim, vec<Real, N>> {
  using type = mat<Real, NewLeftDim, N>;
};
template <size_t NewLeftDim>
struct tensor_add_dimension_left<NewLeftDim, long double> {
  using type = vec<long double, NewLeftDim>;
};
template <size_t NewLeftDim>
struct tensor_add_dimension_left<NewLeftDim, double> {
  using type = vec<double, NewLeftDim>;
};
template <size_t NewLeftDim>
struct tensor_add_dimension_left<NewLeftDim, float> {
  using type = vec<float, NewLeftDim>;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
