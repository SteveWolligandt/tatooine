#ifndef TATOOINE_GRID_VERTEX_ITERATOR_H
#define TATOOINE_GRID_VERTEX_ITERATOR_H
//==============================================================================
#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>

#include "linspace.h"
#include "tensor.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <indexable_space... Dimensions>
class grid;
//==============================================================================
template <indexable_space... Dimensions>
struct grid_vertex_iterator {
  static constexpr auto num_dimensions() { return sizeof...(Dimensions); }
  using difference_type   = size_t;
  using grid_t            = grid<Dimensions...>;
  using value_type        = typename grid_t::pos_t;
  using pointer           = value_type*;
  using reference         = value_type&;
  using iterator_category = std::bidirectional_iterator_tag;
  //============================================================================
  grid_t const* const                  m_grid;
  std::array<size_t, num_dimensions()> m_indices;
  //============================================================================
  template <size_t... Is>
  auto dereference(std::index_sequence<Is...> /*seq*/) const {
    return m_grid->vertex_at(m_indices[Is]...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto dereference() const {
    return dereference(std::make_index_sequence<num_dimensions()>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator*() const { return dereference(); }
  //----------------------------------------------------------------------------
  template <size_t I>
  auto increment_check() {
    if (m_indices[I] == m_grid->template size<I>()) {
      m_indices[I] = 0;
      ++m_indices[I + 1];
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  auto increment(std::index_sequence<Is...> /*seq*/) {
    ++m_indices.front();
    (increment_check<Is>(), ...);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto increment() {
    return increment(std::make_index_sequence<num_dimensions() - 1>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator++() { return increment(); }
  //--------------------------------------------------------------------------
  auto operator==(const grid_vertex_iterator& other) const {
    return m_indices == other.m_indices;
  }
  //--------------------------------------------------------------------------
  auto operator!=(const grid_vertex_iterator& other) const {
    return m_indices != other.m_indices;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<(const grid_vertex_iterator& other) const -> bool {
    return m_indices < other.m_indices;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<=(const grid_vertex_iterator& other) const -> bool {
    return m_indices <= other.m_indices;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>(const grid_vertex_iterator& other) const -> bool {
    return m_indices > other.m_indices;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>=(const grid_vertex_iterator& other) const -> bool {
    return m_indices >= other.m_indices;
  }
};
//==============================================================================
template <indexable_space... Dimensions>
auto next(grid_vertex_iterator<Dimensions...> it, size_t num = 1) {
  for (size_t i = 0; i < num; ++i) { ++it; }
  return it;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
