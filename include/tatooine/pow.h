#ifndef TATOOINE_POW_H
#define TATOOINE_POW_H
//==============================================================================
#include <concepts>
//==============================================================================
namespace tatooine {
//==============================================================================
constexpr auto pow(std::integral auto x, std::unsigned_integral auto const p) {
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
