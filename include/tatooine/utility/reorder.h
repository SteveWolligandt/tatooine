#ifndef TATOOINE_UTILITY_REORDER_H
#define TATOOINE_UTILITY_REORDER_H
//==============================================================================
#include <cassert>
#include <ranges>
//==============================================================================
namespace tatooine {
//==============================================================================
/// reorders a range `data` with another range `order`
auto reorder(std::ranges::range auto& data, std::ranges::range auto& order)
    -> void requires integral<std::ranges::range_value_t<decltype(order)>> {
  assert(std::ranges::size(data) == std::ranges::size(order));

  for (std::size_t vv = 0; vv < size(data) - 1; ++vv) {
    if (order[vv] == vv) {
      continue;
    }
    auto oo = std::size_t{};
    for (oo = vv + 1; oo < order.size(); ++oo) {
      if (order[oo] == vv) {
        break;
      }
    }
    std::swap(data[vv], data[order[vv]]);
    std::swap(order[vv], order[oo]);
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
