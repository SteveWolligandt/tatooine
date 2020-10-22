#ifndef TATOOINE_RANK_H
#define TATOOINE_RANK_H
//==============================================================================
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Tensor>
constexpr auto rank() {
  if constexpr (std::is_arithmetic_v<Tensor>) {
    return 0;
  } else {
    return Tensor::rank();
  }
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor>
constexpr inline auto rank(Tensor &&) {
  return rank<std::decay_t<Tensor>>();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
