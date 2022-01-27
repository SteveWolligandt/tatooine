#ifndef TATOOINE_RENDERING_RENDER_LINE_H
#define TATOOINE_RENDERING_RENDER_LINE_H
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename Real>
auto render_line(vec<Real, 2> x0, vec<Real, 2> x1,
                 uniform_rectilinear_grid<Real, 2> const& grid) {
  auto        nearest_neighbor_grid = uniform_rectilinear_grid<Real, 2>{};
  auto const& ax0                   = grid.template dimension<0>();
  auto        nn_ax0                = ax0;
  nn_ax0.front() -= ax0.spacing() / 2;
  nn_ax0.back() -= ax0.spacing() / 2;
  nn_ax0.push_back();
  auto const& ax1    = grid.template dimension<1>();
  auto        nn_ax1 = ax1;
  nn_ax1.front() -= ax1.spacing() / 2;
  nn_ax1.back() -= ax1.spacing() / 2;
  nn_ax1.push_back();

  auto ix0 = vec<long long, 2>{};
  auto ix1 = vec<long long, 2>{};
  if (x0(0) <= nn_ax0.front()) {
    ix0(0) = 0;
  } else if (x0(0) >= nn_ax0.back()) {
    ix0(0) = size(nn_ax0);
  } else {
    for (std::size_t i = 0; i < size(nn_ax0) - 1; ++i) {
      if (nn_ax0[i] <= x0(0) && x0(0) <= nn_ax0[i + 1]) {
        ix0(0) = i;
        break;
      }
    }
  }
  if (x1(0) <= nn_ax0.front()) {
    ix1(0) = 0;
  } else if (x1(0) >= nn_ax0.back()) {
    ix1(0) = size(nn_ax0);
  } else {
    for (std::size_t i = 0; i < size(nn_ax0) - 1; ++i) {
      if (nn_ax0[i] <= x1(0) && x1(0) <= nn_ax0[i + 1]) {
        ix1(0) = i;
        break;
      }
    }
  }
  if (x0(1) <= nn_ax1.front()) {
    ix0(1) = 0;
  } else if (x0(1) >= nn_ax1.back()) {
    ix0(1) = size(nn_ax1);
  } else {
    for (std::size_t i = 0; i < size(nn_ax1) - 1; ++i) {
      if (nn_ax1[i] <= x0(1) && x0(1) <= nn_ax1[i + 1]) {
        ix0(1) = i;
        break;
      }
    }
  }
  if (x1(1) <= nn_ax1.front()) {
    ix1(1) = 0;
  } else if (x1(1) >= nn_ax1.back()) {
    ix1(1) = size(nn_ax1);
  } else {
    for (std::size_t i = 0; i < size(nn_ax1) - 1; ++i) {
      if (nn_ax1[i] <= x1(1) && x1(1) <= nn_ax1[i + 1]) {
        ix1(1) = i;
        break;
      }
    }
  }

  auto pixels = std::vector<vec<long long, 2>>{};
  auto dx = std::abs<long long>(ix1(0) - ix0(0));
  auto sx = ix0(0) < ix1(0) ? 1 : -1;
  auto dy = -std::abs<long long>(ix1(1) - ix0(1));
  auto sy = ix0(1) < ix1(1) ? 1 : -1;
  auto err = dx + dy;
  auto e2 = std::size_t(0); // error value e_xy
  while (ix0(0) != ix1(0) || ix0(1) != ix1(1)) {
    pixels.push_back(ix0);
    auto const e2 = 2 * err;
    if (e2 > dy) {
      err += dy;
      ix0(0) += sx;
    } // e_xy+e_x > 0
    if (e2 < dx) {
      err += dx;
      ix0(1) += sy;
    } // e_xy+e_y < 0
  }
  pixels.push_back(ix1);
  return pixels;
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
