#ifndef TATOOINE_LINE_VERTEX_ITERATOR_H
#define TATOOINE_LINE_VERTEX_ITERATOR_H
//==============================================================================
#include <boost/iterator/iterator_facade.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N, typename Handle>
struct line_vertex_iterator
    : boost::iterator_facade<line_vertex_iterator<Real, N, Handle>, Handle,
                             boost::bidirectional_traversal_tag, Handle> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = line_vertex_iterator<Real, N, Handle>;

  //============================================================================
  // ctors
  //============================================================================
  line_vertex_iterator(Handle handle) : m_handle{handle} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  line_vertex_iterator(const line_vertex_iterator& other) = default;

  //============================================================================
  // members
  //============================================================================
 private:
  Handle m_handle;

  //============================================================================
  // iterator_facade
  //============================================================================
 private:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  auto increment() -> void { ++m_handle; }
  auto decrement() -> void { --m_handle; }
  //----------------------------------------------------------------------------
  auto equal(const line_vertex_iterator& other) const {
    return m_handle == other.m_handle;
  }
  //----------------------------------------------------------------------------
  auto dereference() const { return m_handle; }

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
template <typename Real, size_t N, typename Handle>
auto next(const line_vertex_iterator<Real, N, Handle>& it, size_t inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle>
auto prev(const line_vertex_iterator<Real, N, Handle>& it, size_t dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle>
auto advance(line_vertex_iterator<Real, N, Handle>& it, size_t inc = 1)
    -> auto& {
  return it.advance(inc);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
