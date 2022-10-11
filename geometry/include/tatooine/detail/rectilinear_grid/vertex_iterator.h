#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_VERTEX_ITERATOR_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_VERTEX_ITERATOR_H
//==============================================================================
#include <tatooine/detail/rectilinear_grid/vertex_handle.h>
#include <tatooine/rectilinear_grid.h>

#include <boost/iterator/iterator_facade.hpp>
#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point_range... Dimensions>
requires(sizeof...(Dimensions) > 1)
class rectilinear_grid;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename... Dimensions>
struct vertex_iterator
    : boost::iterator_facade<vertex_iterator<Dimensions...>,
                             vertex_handle<sizeof...(Dimensions)>,
                             boost::bidirectional_traversal_tag> {
  static constexpr auto num_dimensions() -> std::size_t {
    return sizeof...(Dimensions);
  }
  using difference_type   = size_t;
  using grid_t            = tatooine::rectilinear_grid<Dimensions...>;
  using value_type        = vertex_handle<num_dimensions()>;
  using pointer           = value_type*;
  using reference         = value_type&;
  using iterator_category = std::bidirectional_iterator_tag;
  friend class boost::iterator_core_access;
  //============================================================================
 private:
  grid_t const* const                     m_grid;
  mutable vertex_handle<num_dimensions()> m_handle;
  //============================================================================
 public:
  vertex_iterator(grid_t const* const                    g,
                  vertex_handle<num_dimensions()> const& h)
      : m_grid{g}, m_handle{h} {}
  //----------------------------------------------------------------------------
  vertex_iterator(vertex_iterator const&)     = default;
  vertex_iterator(vertex_iterator&&) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(vertex_iterator const&) -> vertex_iterator& = default;
  auto operator=(vertex_iterator&&) noexcept -> vertex_iterator& = default;
  //============================================================================
 private:
  constexpr auto equal(vertex_iterator const& other) const {
    return m_handle == other.m_handle;
  }
  //----------------------------------------------------------------------------
  constexpr auto dereference() const -> auto& { return m_handle; }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto decrement_check(bool& stop) {
    if (!stop && m_handle.indices()[I] == 0) {
      m_handle.indices()[I] = m_grid->template size<I>() - 1;
    } else {
      --m_handle.indices()[I];
      stop = true;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  constexpr auto decrement(std::index_sequence<Is...> /*seq*/) {
    --m_handle.plain_index();
    bool stop = false;
    (decrement_check<Is>(stop), ...);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto decrement() {
    return decrement(std::make_index_sequence<num_dimensions() - 1>{});
  }
  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto increment_check(bool& stop) {
    if (!stop && m_handle.indices()[I] == m_grid->template size<I>()) {
      m_handle.indices()[I] = 0;
      ++m_handle.indices()[I + 1];
    } else {
      stop = true;
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t... Is>
  constexpr auto increment(std::index_sequence<Is...> /*seq*/) {
    ++m_handle.plain_index();
    ++m_handle.indices().front();
    bool stop = false;
    (increment_check<Is>(stop), ...);
    return *this;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto increment() {
    return increment(std::make_index_sequence<num_dimensions() - 1>{});
  }

 public:
  //--------------------------------------------------------------------------
  constexpr auto operator<(vertex_iterator const& other) const -> bool {
    return m_handle < other.m_handle;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<=(vertex_iterator const& other) const -> bool {
    return m_handle <= other.m_handle;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>(vertex_iterator const& other) const -> bool {
    return m_handle > other.m_handle;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>=(vertex_iterator const& other) const -> bool {
    return m_handle >= other.m_handle;
  }
};
//==============================================================================
template <dimension... Dimensions>
auto next(vertex_iterator<Dimensions...> it, size_t num = 1) {
  for (size_t i = 0; i < num; ++i) {
    ++it;
  }
  return it;
}
//==============================================================================
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
#endif
