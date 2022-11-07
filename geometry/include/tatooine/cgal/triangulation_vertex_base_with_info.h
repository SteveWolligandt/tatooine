#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_VERTEX_BASE_WITH_INFO_H
#define TATOOINE_CGAL_VERTEX_BASE_WITH_INFO_H
//==============================================================================
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <tatooine/cgal/triangulation_vertex_base.h>
//==============================================================================
namespace tatooine::cgal {
//==============================================================================
/// \defgroup cgal_triangulation_vertex_base_with_info Triangulation vertex base
/// with info \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename Info, typename Traits,
          typename VertexBase>
struct triangulation_vertex_base_with_info_impl;
//------------------------------------------------------------------------------
template <typename Info, typename Traits, typename VertexBase>
struct triangulation_vertex_base_with_info_impl<2, Info, Traits, VertexBase> {
  using type =
      CGAL::Triangulation_vertex_base_with_info_2<Info, Traits, VertexBase>;
};
//------------------------------------------------------------------------------
template <typename Info, typename Traits, typename VertexBase>
struct triangulation_vertex_base_with_info_impl<3, Info, Traits, VertexBase> {
  using type =
      CGAL::Triangulation_vertex_base_with_info_3<Info, Traits, VertexBase>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Info, typename Traits,
          typename VertexBase =
              triangulation_vertex_base<NumDimensions, Traits>>
using triangulation_vertex_base_with_info =
    typename triangulation_vertex_base_with_info_impl<NumDimensions, Info,
                                                      Traits, VertexBase>::type;
/// \}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
