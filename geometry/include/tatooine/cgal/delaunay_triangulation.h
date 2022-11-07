#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_DELAUNAY_TRIANGULATION_H
#define TATOOINE_CGAL_DELAUNAY_TRIANGULATION_H
//==============================================================================
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_with_circumcenter_3.h>
#include <tatooine/cgal/triangulation_data_structure.h>
#include <tatooine/cgal/triangulation_simplex_base.h>
#include <tatooine/cgal/triangulation_vertex_base.h>
#include <tatooine/cgal/triangulation_vertex_base_with_info.h>
//==============================================================================
namespace tatooine::cgal {
/// \defgroup cgal_delaunay_triangulation Delaunay Triangulation
/// \ingroup cgal
/// \{
//==============================================================================
template <std::size_t NumDimensions, typename Traits,
          typename TriangulationDataStructure>
struct delaunay_triangulation_impl;
//------------------------------------------------------------------------------
template <typename Traits, typename TriangulationDataStructure>
struct delaunay_triangulation_impl<2, Traits, TriangulationDataStructure> {
  using type =
      CGAL::Delaunay_triangulation_2<Traits, TriangulationDataStructure>;
};
//------------------------------------------------------------------------------
template <typename Traits, typename TriangulationDataStructure>
struct delaunay_triangulation_impl<3, Traits, TriangulationDataStructure> {
  using type =
      CGAL::Delaunay_triangulation_3<Traits, TriangulationDataStructure>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Traits,
          typename TriangulationDataStructure =
              triangulation_data_structure<NumDimensions>>
using delaunay_triangulation =
    typename delaunay_triangulation_impl<NumDimensions, Traits,
                                         TriangulationDataStructure>::type;
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Traits, typename Info,
          typename SimplexBase = triangulation_ds_simplex_base<NumDimensions>>
using delaunay_triangulation_with_info = delaunay_triangulation<
    NumDimensions, Traits,
    triangulation_data_structure<
        NumDimensions,
        triangulation_vertex_base_with_info<NumDimensions, Info, Traits>,
        SimplexBase>>;
//==============================================================================
/// \}
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
