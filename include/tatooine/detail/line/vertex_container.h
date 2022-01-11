#ifndef TATOOINE_DETAIL_LINE_VERTEX_CONTAINER_H
#define TATOOINE_DETAIL_LINE_VERTEX_CONTAINER_H
//==============================================================================
#include <tatooine/detail/line/vertex_iterator.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
struct line;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::detail::line {
//============================================================================
template <typename Real, std::size_t NumDimensions, typename Handle>
struct vertex_container {
  //============================================================================
  // typedefs
  //============================================================================
  using iterator  = vertex_iterator<Real, NumDimensions, Handle>;
  using line_type = tatooine::line<Real, NumDimensions>;

  //============================================================================
  // members
  //============================================================================
  line_type const& m_line;

  //============================================================================
  // methods
  //============================================================================
  auto begin() const { return iterator{Handle{0}}; }
  auto end() const { return iterator{Handle{m_line.m_vertices.size()}}; }
  auto front() const { return Handle{0}; }
  auto back() const { return Handle{m_line.m_vertices.size() - 1}; }
  auto size() const { return m_line.m_vertices.size(); }
  auto data_container() const -> auto const& { return m_line.m_vertices; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, std::size_t NumDimensions, typename Handle>
auto begin(vertex_container<Real, NumDimensions, Handle> const& it) {
  return it.begin();
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, std::size_t NumDimensions, typename Handle>
auto end(vertex_container<Real, NumDimensions, Handle> const& it) {
  return it.begin();
}
//==============================================================================
}  // namespace tatooine::detail::line
//==============================================================================
#endif
