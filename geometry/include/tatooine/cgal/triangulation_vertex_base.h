#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_VERTEX_BASE_H
#define TATOOINE_CGAL_VERTEX_BASE_H
//==============================================================================
#include <tatooine/cgal/triangulation_ds_vertex_base.h>
//==============================================================================
namespace tatooine::cgal {
//==============================================================================
/// \defgroup cgal_triangulation_vertex_base Triangulation Vertex Base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename Traits, typename VertexBase>
struct triangulation_vertex_base_impl;
//------------------------------------------------------------------------------
template <typename Traits, typename VertexBase>
struct triangulation_vertex_base_impl<2, Traits, VertexBase> {
  using type = CGAL::Triangulation_vertex_base_2<Traits, VertexBase>;
};
//------------------------------------------------------------------------------
template <typename Traits, typename VertexBase>
struct triangulation_vertex_base_impl<3, Traits, VertexBase> {
  using type = CGAL::Triangulation_vertex_base_3<Traits, VertexBase>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Traits,
          typename VertexBase = triangulation_ds_vertex_base<NumDimensions>>
using triangulation_vertex_base =
    typename triangulation_vertex_base_impl<NumDimensions, Traits,
                                            VertexBase>::type;
/// \}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
