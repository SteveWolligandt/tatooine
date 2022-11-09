#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_TRIANGULATION_DS_VERTEX_BASE_H
#define TATOOINE_CGAL_TRIANGULATION_DS_VERTEX_BASE_H
//==============================================================================
#include <CGAL/Triangulation_ds_vertex_base_2.h>
#include <CGAL/Triangulation_ds_vertex_base_3.h>
//==============================================================================
namespace tatooine::cgal {
//==============================================================================
/// \defgroup cgal_triangulation_ds_vertex_base Triangulation DS Vertex Base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename TDS>
struct triangulation_ds_vertex_base_impl;
//------------------------------------------------------------------------------
template <typename TDS>
struct triangulation_ds_vertex_base_impl<2, TDS> {
  using type = CGAL::Triangulation_ds_vertex_base_2<TDS>;
};
//------------------------------------------------------------------------------
template <typename TDS>
struct triangulation_ds_vertex_base_impl<3, TDS> {
  using type = CGAL::Triangulation_ds_vertex_base_3<TDS>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename TDS = void>
using triangulation_ds_vertex_base =
    typename triangulation_ds_vertex_base_impl<NumDimensions, TDS>::type;
/// \}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
