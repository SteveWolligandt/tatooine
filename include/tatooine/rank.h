#ifndef TATOOINE_RANK_H
#define TATOOINE_RANK_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex Scalar>
constexpr auto rank() {
  return 0;
}
template <arithmetic_or_complex Scalar>
constexpr auto rank(Scalar&&) {
  return 0;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_tensor Tensor>
constexpr auto rank() {
  return std::decay_t<Tensor>::rank();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <static_tensor Tensor>
constexpr auto rank(Tensor &&) { return std::decay_t<Tensor>::rank(); }
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <dynamic_tensor Tensor>
constexpr auto rank(Tensor && t) { return t.rank(); }
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
