#ifndef TATOOINE_POW_H
#define TATOOINE_POW_H
//==============================================================================
#include <concepts>
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::integral Int>
constexpr auto pow(Int x, std::unsigned_integral auto const p) -> Int {
  if (p == 0) {
    return 1;
  }
  if (p == 1) {
    return x;
  }

  auto const tmp = pow(x, p / 2);
  if (p % 2 == 0) {
    return tmp * tmp;
  } else {
    return x * tmp * tmp;
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
