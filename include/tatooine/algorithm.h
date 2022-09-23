#ifndef TATOOINE_ALGORITHM_H
#define TATOOINE_ALGORITHM_H
//==============================================================================
#include <cstddef>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Range, typename RangeIt>
decltype(auto) resize_prev_list(Range& range, RangeIt pos,
                                std::size_t new_prev_size) {
  auto prev_size = distance(begin(range), pos);
  auto size_change =
      static_cast<int>(new_prev_size) - static_cast<int>(prev_size);
  if (size_change > 0) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(size_change); ++i) {
      range.emplace_front(range.front());
    }
  } else if (size_change < 0) {
    range.erase(begin(range), next(begin(range), prev_size - new_prev_size));
  }
  return range;
}
//------------------------------------------------------------------------------
template <typename Range, typename RangeIt>
decltype(auto) resize_next_list(Range& range, RangeIt pos,
                                std::size_t new_next_size) {
  auto next_size = distance(next(pos), end(range));
  int  size_change =
      static_cast<int>(new_next_size) - static_cast<int>(next_size);
  if (size_change > 0) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(size_change); ++i) {
      range.emplace_back(range.back());
    }
  } else if (size_change < 0) {
    range.erase(prev(end(range), next_size - new_next_size), end(range));
  }
}
//------------------------------------------------------------------------------
template <typename T>
constexpr auto clamp(const T& v, const T& lo, const T& hi) -> const T& {
  assert(!(hi < lo));
  return (v < lo) ? lo : (hi < v) ? hi : v;
}
//------------------------------------------------------------------------------
template <typename T, typename Compare>
constexpr auto clamp(const T& v, const T& lo, const T& hi, Compare comp)
    -> const T& {
  assert(!comp(hi, lo));
  return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
