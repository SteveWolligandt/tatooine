#ifndef TATOOINE_DETAIL_LINE_VERTEX_ITERATOR_H
#define TATOOINE_DETAIL_LINE_VERTEX_ITERATOR_H
//==============================================================================
#include <boost/iterator/iterator_facade.hpp>
//==============================================================================
namespace tatooine::detail::line {
//==============================================================================
template <typename Real, size_t NumDimensions, typename Handle>
struct vertex_iterator
    : boost::iterator_facade<vertex_iterator<Real, NumDimensions, Handle>,
                             Handle, boost::bidirectional_traversal_tag,
                             Handle> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t = vertex_iterator<Real, NumDimensions, Handle>;

  //============================================================================
  // ctors
  //============================================================================
  vertex_iterator(Handle handle) : m_handle{handle} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  vertex_iterator(const vertex_iterator& other) = default;

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
  auto equal(const vertex_iterator& other) const {
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
template <typename Real, size_t NumDimensions, typename Handle>
auto next(const vertex_iterator<Real, NumDimensions, Handle>& it,
          size_t                                              inc = 1) {
  return it.next(inc);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t NumDimensions, typename Handle>
auto prev(const vertex_iterator<Real, NumDimensions, Handle>& it,
          size_t                                              dec = 1) {
  return it.prev(dec);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t NumDimensions, typename Handle>
auto advance(vertex_iterator<Real, NumDimensions, Handle>& it, size_t inc = 1)
    -> auto& {
  return it.advance(inc);
}
//==============================================================================
}  // namespace tatooine::detail::line
//==============================================================================
#endif
