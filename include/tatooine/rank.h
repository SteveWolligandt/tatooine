#ifndef TATOOINE_RANK_H
#define TATOOINE_RANK_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number Tensor>
constexpr auto rank() {
  return 0;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor>
requires is_tensor_v<Tensor>
constexpr auto rank(Tensor &&) {
  return rank<std::decay_t<Tensor>>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
