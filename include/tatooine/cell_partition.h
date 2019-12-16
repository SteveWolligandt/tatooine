#ifndef TATOOINE_CELL_PARTITION_H
#define TATOOINE_CELL_PARTITION_H

#include "grid.h"
#include <omp.h>

//==============================================================================
namespace tatooine {
//==============================================================================
/// iterates over all cells of the grid in parallel
template <typename F, typename GridReal>
auto cell_partition(F&& f, const grid<GridReal, 3>& g) {
  using vec3 = vec<Real, 3>;
#ifdef NDEBUG
  omp_lock_t lock;
  omp_init_lock(&lock);
#pragma omp parallel for collapse(3)
#endif
  for (size_t iz = 0; iz < g.dimension(2).size() - 1; ++iz) {
    for (size_t iy = 0; iy < g.dimension(1).size() - 1; ++iy) {
      for (size_t ix = 0; ix < g.dimension(0).size() - 1; ++ix) {
        const auto& x0 = g.dimension(0)[ix];
        const auto& x1 = g.dimension(0)[ix + 1];
        const auto& y0 = g.dimension(1)[iy];
        const auto& y1 = g.dimension(1)[iy + 1];
        const auto& z0 = g.dimension(2)[iz];
        const auto& z1 = g.dimension(2)[iz + 1];
        f(ix, iy, iz,
          std::array{
              vec{x0, y0, z0},
              vec{x1, y0, z0},
              vec{x0, y1, z0},
              vec{x1, y1, z0},
              vec{x0, y0, z1},
              vec{x1, y0, z1},
              vec{x0, y1, z1},
              vec{x1, y1, z1},
          }, lock);
      }
    }
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
