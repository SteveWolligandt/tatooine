#ifndef TATOOINE_TENSOR_CAST_H
#define TATOOINE_TENSOR_CAST_H
//==============================================================================
#include <tatooine/tensor.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename CastReal, typename Real, size_t N>
auto cast_tensor_type_impl(const vec<Real, N>&) {
  return vec<CastReal, N>::zeros();
}
template <typename CastReal, typename Real, size_t M, size_t N>
auto cast_tensor_type_impl(const mat<Real, M, N>&) {
  return mat<CastReal, M, N>::zeros();
}
template <typename CastReal, typename Real, size_t... Dims>
auto cast_tensor_type_impl(const tensor<Real, Dims...>&) {
  return tensor<CastReal, Dims...>::zeros();
}

template <typename CastedReal, typename Tensor>
struct cast_tensor_real {
  using type =
      decltype(cast_tensor_type_impl<CastedReal>(std::declval<Tensor>()));
};

template <typename CastedReal, typename Tensor>
using cast_tensor_real_t = typename cast_tensor_real<CastedReal, Tensor>::type;

//==============================================================================
template <typename NewReal, typename Tensor, typename Real, size_t... Dims>
auto cast(const base_tensor<Tensor, Real, Dims...>& to_cast) {
  auto casted = tensor<NewReal, Dims...>::zeros;
  for (size_t i = 0; i < casted.num_components(); ++i) {
    casted[i] = static_cast<NewReal>(to_cast[i]);
  }
  return casted;
}
//------------------------------------------------------------------------------
template <typename NewReal, typename Real, size_t M, size_t N>
auto cast(const mat<Real, M, N>& to_cast) {
  auto casted = mat<NewReal, M, N>::zeros();
  for (size_t i = 0; i < M * N; ++i) {
    casted[i] = static_cast<NewReal>(to_cast[i]);
  }
  return casted;
}
//------------------------------------------------------------------------------
template <typename NewReal, typename Real, size_t N>
auto cast(const vec<Real, N>& to_cast) {
  auto casted = vec<NewReal, N>::zeros();
  for (size_t i = 0; i < N; ++i) {
    casted[i] = static_cast<NewReal>(to_cast[i]);
  }
  return casted;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
