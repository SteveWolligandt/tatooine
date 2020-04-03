#ifndef TATOOINE_TENSOR_TYPE_H
#define TATOOINE_TENSOR_TYPE_H
//╔════════════════════════════════════════════════════════════════════════════╗
#include "tensor.h"
#include "type_traits.h"
//╔════════════════════════════════════════════════════════════════════════════╗
namespace tatooine {
template <typename Real, size_t... Dims>
struct tensor_type_impl {
  using type = tensor<Real, Dims...>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N>
struct tensor_type_impl<Real, N> {
  using type = vec<Real, N>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t M, size_t N>
struct tensor_type_impl<Real, M, N> {
  using type = mat<Real, M, N>;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t... Dims>
using tensor_type = typename tensor_type_impl<Real, Dims...>::type;
}  // namespace tatooine
//╚════════════════════════════════════════════════════════════════════════════╝
#endif
