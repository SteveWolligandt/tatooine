#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_TRIANGULATION_DATA_STRUCTURE_H
#define TATOOINE_CGAL_TRIANGULATION_DATA_STRUCTURE_H
//==============================================================================
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <tatooine/cgal/triangulation_ds_simplex_base.h>
#include <tatooine/cgal/triangulation_ds_vertex_base.h>
//==============================================================================
namespace tatooine::cgal {
//==============================================================================
/// \defgroup cgal_triangulation_data_structure Triangulation data structure
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename VertexBase, typename SimplexBase>
struct triangulation_data_structure_impl;
//------------------------------------------------------------------------------
template <typename VertexBase, typename SimplexBase>
struct triangulation_data_structure_impl<2, VertexBase, SimplexBase> {
  using type = CGAL::Triangulation_data_structure_2<VertexBase, SimplexBase>;
};
//------------------------------------------------------------------------------
template <typename VertexBase, typename SimplexBase>
struct triangulation_data_structure_impl<3, VertexBase, SimplexBase> {
  using type = CGAL::Triangulation_data_structure_3<VertexBase, SimplexBase>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions,
          typename VertexBase  = triangulation_ds_vertex_base<NumDimensions>,
          typename SimplexBase = triangulation_ds_simplex_base<NumDimensions>>
using triangulation_data_structure =
    typename triangulation_data_structure_impl<NumDimensions, VertexBase,
                                               SimplexBase>::type;
/// \}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
