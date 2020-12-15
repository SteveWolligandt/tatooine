#ifndef TATOOINE_RANK_H
#define TATOOINE_RANK_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number Scalar>
constexpr auto rank() {
  return 0;
}
template <real_number Scalar>
constexpr auto rank(Scalar&&) {
  return 0;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor>
requires is_tensor_v<Tensor>
constexpr auto rank() {
  return std::decay_t<Tensor>::rank();
}
template <typename Tensor>
requires is_tensor_v<Tensor>
constexpr auto rank(Tensor &&) {
  return std::decay_t<Tensor>::rank();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
