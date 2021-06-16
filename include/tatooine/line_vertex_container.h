#ifndef TATOOINE_LINE_VERTEX_CONTAINER_H
#define TATOOINE_LINE_VERTEX_CONTAINER_H
//==============================================================================
#include <tatooine/line_vertex_iterator.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct line;
//============================================================================
template <typename Real, size_t N, typename Handle>
struct line_vertex_container {
  //============================================================================
  // typedefs
  //============================================================================
  using iterator       = line_vertex_iterator<Real, N, Handle>;

  //============================================================================
  // members
  //============================================================================
  line<Real, N> const& m_line;

  //============================================================================
  // methods
  //============================================================================
  auto begin() const { return iterator{Handle{0}}; }
  auto end()   const { return iterator{Handle{m_line.num_vertices()}}; }
  auto front() const { return Handle{0}; }
  auto back()  const { return Handle{m_line.num_vertices() - 1}; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, size_t N, typename Handle>
auto begin(line_vertex_container<Real, N, Handle> const& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t N, typename Handle>
auto end(line_vertex_container<Real, N, Handle> const& it) {
  return it.begin();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
