#ifndef TATOOINE_RANK_H
#define TATOOINE_RANK_H
//==============================================================================
#ifdef __cpp_concepts
#include <tatooine/concepts.h>
#endif
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_concepts
template <arithmetic_or_complex Scalar>
#else
template <typename Scalar, enable_if<is_arithmetic_or_complex<Scalar>> = true>
#endif
constexpr auto rank() {
  return 0;
}
#ifdef __cpp_concepts
template <arithmetic_or_complex Scalar>
#else
template <typename Scalar, enable_if<is_arithmetic_or_complex<Scalar>> = true>
#endif
constexpr auto rank(Scalar&&) {
  return 0;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
template <typename Tensor>
requires is_tensor<Tensor>
#else
template <typename Tensor, enable_if_tensor<Tensor> = true>
#endif
constexpr auto rank() {
  return std::decay_t<Tensor>::rank();
}
#ifdef __cpp_concepts
template <typename Tensor>
requires is_tensor<Tensor>
#else
template <typename Tensor, enable_if_tensor<Tensor> = true>
#endif
constexpr auto rank(Tensor &&) {
  return std::decay_t<Tensor>::rank();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
