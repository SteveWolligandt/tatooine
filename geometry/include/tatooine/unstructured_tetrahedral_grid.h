#ifndef TATOOINE_UNSTRUCTURED_TETRAHEDRAL_GRID_H
#define TATOOINE_UNSTRUCTURED_TETRAHEDRAL_GRID_H
//==============================================================================
//#if TATOOINE_CGAL_AVAILABLE
//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Delaunay_triangulation_3.h>
//#include <CGAL/Triangulation_vertex_base_with_info_3.h>
//#endif
//
//#include <tatooine/rectilinear_grid.h>
//#include <tatooine/pointset.h>
//#include <tatooine/property.h>
//#include <tatooine/vtk_legacy.h>
//
//#include <boost/range/algorithm/copy.hpp>
//#include <boost/range/adaptor/transformed.hpp>
//#include <vector>
#include <tatooine/unstructured_simplicial_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
using unstructured_tetrahedral_grid = unstructured_simplicial_grid<Real, N, 3>;
using unstructured_tetrahedral_grid3 =
    unstructured_tetrahedral_grid<real_number, 3>;
using unstructured_tetrahedral_grid4 =
    unstructured_tetrahedral_grid<real_number, 4>;
using unstructured_tetrahedral_grid5 =
    unstructured_tetrahedral_grid<real_number, 5>;
using unstructured_tetrahedral_grid6 =
    unstructured_tetrahedral_grid<real_number, 6>;
using unstructured_tetrahedral_grid7 =
    unstructured_tetrahedral_grid<real_number, 7>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif

