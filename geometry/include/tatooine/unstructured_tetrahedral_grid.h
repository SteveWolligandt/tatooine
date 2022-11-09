#ifndef TATOOINE_UNSTRUCTURED_TETRAHEDRAL_GRID_H
#define TATOOINE_UNSTRUCTURED_TETRAHEDRAL_GRID_H
//==============================================================================
#include <tatooine/unstructured_simplicial_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t N>
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

