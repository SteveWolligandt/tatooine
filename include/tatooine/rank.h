#ifndef TATOOINE_RANK_H
#define TATOOINE_RANK_H
//==============================================================================
#include <tatooine/base_tensor.h>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_or_complex_number Scalar>
constexpr auto rank() {
  return 0;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_or_complex_number Scalar>
constexpr auto rank(Scalar const&) {
  return 0;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T> struct is_tensor;
template <typename Tensor>
requires is_tensor<Tensor>::value
constexpr auto rank() {
  return Tensor::rank();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, real_or_complex_number T, size_t... Dims>
struct base_tensor;
template <typename Tensor, real_or_complex_number T, size_t... Dims>
constexpr auto rank(base_tensor<Tensor, T, Dims...> const&) {
  return sizeof...(Dims);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
