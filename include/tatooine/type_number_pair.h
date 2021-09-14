#ifndef TATOOINE_TYPE_NUMBER_PAIR_H
#define TATOOINE_TYPE_NUMBER_PAIR_H
//==============================================================================
#include <cstdint>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, std::size_t N>
struct type_number_pair {
  using type = T;
  static auto constexpr value = N;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
