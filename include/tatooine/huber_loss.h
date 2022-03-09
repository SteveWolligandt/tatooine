#ifndef TATOOINE_HUBER_LOSS_H
#define TATOOINE_HUBER_LOSS_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/math.h>
#include <tatooine/real.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// See <a href = "https://en.wikipedia.org/wiki/Huber_loss">Wikipedia</a>
template <arithmetic Delta = int>
auto constexpr huber_loss(floating_point auto const a, Delta const delta = 1) {
  if (gcem::abs(a) <= delta) {
    return a * a / 2;
  }
  return delta * (gcem::abs(a) - delta / 2);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
