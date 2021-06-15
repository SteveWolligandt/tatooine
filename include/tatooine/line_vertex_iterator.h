#ifndef TATOOINE_LINE_VERTEX_ITERATOR_H
#define TATOOINE_LINE_VERTEX_ITERATOR_H
//==============================================================================
#include <boost/iterator/iterator_facade.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct line;
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
struct const_line_vertex_iterator
    : boost::iterator_facade<
          const_line_vertex_iterator<Line, Real, N, Handle, Value>,
          Value, boost::bidirectional_traversal_tag> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t =
      const_line_vertex_iterator<Line, Real, N, Handle, Value>;

  //============================================================================
  // ctors
  //============================================================================
  const_line_vertex_iterator(Handle handle, const Line& l, bool prefer_calc)
      : m_handle{handle}, m_line{l}, m_prefer_calc{prefer_calc} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  const_line_vertex_iterator(const const_line_vertex_iterator& other) = default;

  //============================================================================
  // members
  //============================================================================
 private:
  Handle      m_handle;
  const Line& m_line;
  bool        m_prefer_calc;

  //============================================================================
  // iterator_face
  //============================================================================
 private:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  auto increment() -> void { ++m_handle; }
  auto decrement() -> void { --m_handle; }
  //----------------------------------------------------------------------------
  auto equal(const const_line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  //----------------------------------------------------------------------------
  auto dereference() const -> decltype(auto) {
    return m_line.at(m_handle, m_prefer_calc);
  }
  //============================================================================
  // methods
  //============================================================================
 public:
  auto next(size_t inc = 1) const -> this_t{
    this_t n = *this;
    n.m_handle.i += inc;
    return n;
  }
  //----------------------------------------------------------------------------
  auto prev(size_t dec = 1) const -> this_t{
    this_t p = *this;
    p.m_handle.i -= dec;
    return p;
  }
  //----------------------------------------------------------------------------
  auto advance(const size_t inc = 1) const -> auto& {
    m_handle.i += inc;
    return *this;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto next(const const_line_vertex_iterator<Line, Real, N, Handle, Value>& it,
                 size_t                                       inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto prev(const const_line_vertex_iterator<Line, Real, N, Handle, Value>& it,
          size_t                                       dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto advance(
    const_line_vertex_iterator<Line, Real, N, Handle, Value>& it,
    size_t inc = 1)->auto& {
  return it.advance(inc);
}
//==============================================================================
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
struct line_vertex_iterator
    : boost::iterator_facade<line_vertex_iterator<Line, Real, N, Handle, Value>,
                             Value, boost::bidirectional_traversal_tag> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = line_vertex_iterator<Line, Real, N, Handle, Value>;

  //============================================================================
  // ctors
  //============================================================================
  line_vertex_iterator(Handle handle, Line& l) : m_handle{handle}, m_line{l} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line_vertex_iterator(const line_vertex_iterator& other) = default;

  //============================================================================
  // members
  //============================================================================
 private:
  Handle m_handle;
  Line&  m_line;
  bool   m_prefer_calc;

  //============================================================================
  // iterator_facade
  //============================================================================
 private:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  auto increment() ->void{ ++m_handle; }
  auto decrement() ->void{ --m_handle; }
  //----------------------------------------------------------------------------
  auto equal(const line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  //----------------------------------------------------------------------------
  auto dereference() -> decltype(auto) { return m_line.at(m_handle, m_prefer_calc); }

  //============================================================================
  // methods
  //============================================================================
 public:
  auto next(const size_t inc = 1) const -> this_t {
    this_t n = *this;
    n.m_handle.i += inc;
    return n;
  }
  auto prev(const size_t dec = 1) const -> this_t {
    this_t p = *this;
    p.m_handle.i -= dec;
    return p;
  }
  auto advance(const size_t inc = 1) const -> auto& {
    m_handle.i += inc;
    return *this;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto next(
    const line_vertex_iterator<Line, Real, N, Handle, Value>& it,
    size_t inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto prev(
    const line_vertex_iterator<Line, Real, N, Handle, Value>& it,
    size_t dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Line, typename Real, size_t N, typename Handle,
          typename Value>
auto advance(line_vertex_iterator<Line, Real, N, Handle, Value>& it,
             size_t inc = 1) -> auto& {
  return it.advance(inc);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
