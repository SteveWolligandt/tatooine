#ifndef TATOOINE_LINE_VERTEX_CONTAINER_H
#define TATOOINE_LINE_VERTEX_CONTAINER_H
//==============================================================================
#include <tatooine/line_vertex_iterator.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct line;
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
struct const_line_vertex_container {
  //============================================================================
  // typedefs
  //============================================================================
  using iterator =
      line_vertex_iterator<Line, Real, N, Handle, Value>;
  using const_iterator =
      const_line_vertex_iterator<Line, Real, N, Handle, Value>;

  //============================================================================
  // members
  //============================================================================
  Line const& m_line;

  //============================================================================
  // methods
  //============================================================================
  auto begin() const {
    return const_iterator{Handle{0}, m_line};
  }
  // -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() const {
    return const_iterator{Handle{m_line.num_vertices()}, m_line};
  }
  //--------------------------------------------------------------------------
  auto front() const -> auto const& {
    return m_line.at(Handle{0});
  }
  //--------------------------------------------------------------------------
  auto back() const -> auto const& {
    return m_line.at(Handle{m_line.num_vertices() - 1});
  }
  //--------------------------------------------------------------------------
  auto line() const -> auto const& { return m_line; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto begin(const_line_vertex_container<Line, Real, N, Handle, Value> const& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto end(const_line_vertex_container<Line, Real, N, Handle, Value> const& it) {
  return it.begin();
}
//============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
struct line_vertex_container {
  //============================================================================
  // typedefs
  //============================================================================
  using iterator =
      line_vertex_iterator<Line, Real, N, Handle, Value>;
  using const_iterator =
      const_line_vertex_iterator<Line, Real, N, Handle, Value>;

  //============================================================================
  // members
  //============================================================================
  Line& m_line;

  //============================================================================
  // methods
  //============================================================================
  auto begin() const {
    return const_iterator{Handle{0}, m_line};
  }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto begin() { return iterator{Handle{0}, m_line}; }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() const {
    return const_iterator{Handle{m_line.num_vertices()}, m_line};
  }
  //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
  auto end() {
    return iterator{Handle{m_line.num_vertices()}, m_line};
  }
  //----------------------------------------------------------------------------
  auto front() const -> auto const&{ return m_line.at(Handle{0}); }
  auto front() -> auto& { return m_line.at(Handle{0}); }
  //----------------------------------------------------------------------------
  auto back() const -> auto const&{
    return m_line.at(Handle{m_line.num_vertices() - 1});
  }
  auto back() -> auto& {
    return m_line.at(Handle{m_line.num_vertices() - 1});
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto begin(
    line_vertex_container<Line, Real, N, Handle, Value> const& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto begin(line_vertex_container<Line, Real, N, Handle, Value>& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto end(
    line_vertex_container<Line, Real, N, Handle, Value> const& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto end(line_vertex_container<Line, Real, N, Handle, Value>& it) {
  return it.begin();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
