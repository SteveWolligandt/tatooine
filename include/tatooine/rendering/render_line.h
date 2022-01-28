#ifndef TATOOINE_RENDERING_RENDER_LINE_H
#define TATOOINE_RENDERING_RENDER_LINE_H
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename Real>
auto render_line(vec<Real, 2> p0, vec<Real, 2> p1,
                 uniform_rectilinear_grid<Real, 2> const& grid) {
  auto        pixels                = std::vector<vec<long long, 2>>{};
  auto        nearest_neighbor_grid = uniform_rectilinear_grid<Real, 2>{};
  auto const& ax_x                  = grid.template dimension<0>();
  auto        nn_ax_x               = ax_x;
  nn_ax_x.front() -= ax_x.spacing() / 2;
  nn_ax_x.back() -= ax_x.spacing() / 2;
  nn_ax_x.push_back();
  auto const& ax_y    = grid.template dimension<1>();
  auto        nn_ax_y = ax_y;
  nn_ax_y.front() -= ax_y.spacing() / 2;
  nn_ax_y.back() -= ax_y.spacing() / 2;
  nn_ax_y.push_back();

  if ((p0.x() < nn_ax_x.front() && p1.x() < nn_ax_x.front()) ||
      (p0.x() > nn_ax_x.back() && p1.x() > nn_ax_x.back()) ||
      (p0.y() < nn_ax_y.front() && p1.y() < nn_ax_y.front()) ||
      (p0.y() > nn_ax_y.back() && p1.y() > nn_ax_y.back())) {
    return pixels;
  }

  auto get_t = [](auto const a, auto const b, auto const c) {
    return (c - a) / (b - a);
  };
  // clamp p0.x()
  if (p0.x() < nn_ax_x.front()) {
    auto const t = get_t(p0.x(), p1.x(), nn_ax_x.front());
    p0           = vec{nn_ax_x.front(), p0.y() * (1 - t) + p1.y() * t};
  } else if (p0.x() > nn_ax_x.back()) {
    auto const t =  get_t(p0.x(), p1.x(), nn_ax_x.back());
    p0           = vec{nn_ax_x.back(), p0.y() * (1 - t) + p1.y() * t};
  }
  // clamp p1.x()
  if (p1.x() <= nn_ax_x.front()) {
    auto const t =  get_t(p0.x(), p1.x(), nn_ax_x.front());
    p1           = vec{nn_ax_x.front(), p0.y() * (1 - t) + p1.y() * t};
  } else if (p1.x() > nn_ax_x.back()) {
    auto const t =  get_t(p0.x(), p1.x(), nn_ax_x.back());
    p1           = vec{nn_ax_x.back(), p0.y() * (1 - t) + p1.y() * t};
  }
  // clamp p0.y()
  if (p0.y() < nn_ax_y.front()) {
    auto const t =  get_t(p0.y(), p1.y(), nn_ax_y.front());
    p0           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.front()};
  } else if (p0.y() > nn_ax_y.back()) {
    auto const t =  get_t(p0.y(), p1.y(), nn_ax_y.back());
    p0           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.back()};
  }
  // clamp p1.y()
  if (p1.y() < nn_ax_y.front()) {
    auto const t =  get_t(p0.y(), p1.y(), nn_ax_y.front());
    p1           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.front()};
  } else if (p1.y() > nn_ax_y.back()) {
    auto const t =  get_t(p0.y(), p1.y(), nn_ax_y.back());
    p1           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.back()};
  }
  auto ip0 = vec<long long, 2>{};
  auto ip1 = vec<long long, 2>{};
  // find ip0.x
  for (std::size_t i = 0; i < size(nn_ax_x) - 1; ++i) {
    if (nn_ax_x[i] <= p0.x() && p0.x() <= nn_ax_x[i + 1]) {
      ip0.x() = i;
      break;
    }
  }
  // find ip1.x
  for (std::size_t i = 0; i < size(nn_ax_x) - 1; ++i) {
    if (nn_ax_x[i] <= p1.x() && p1.x() <= nn_ax_x[i + 1]) {
      ip1.x() = i;
      break;
    }
  }
  // find ip0.y
  for (std::size_t i = 0; i < size(nn_ax_y) - 1; ++i) {
    if (nn_ax_y[i] <= p0.y() && p0.y() <= nn_ax_y[i + 1]) {
      ip0.y() = i;
      break;
    }
  }
  // find ip1.y
  for (std::size_t i = 0; i < size(nn_ax_y) - 1; ++i) {
    if (nn_ax_y[i] <= p1.y() && p1.y() <= nn_ax_y[i + 1]) {
      ip1.y() = i;
      break;
    }
  }

  //if (ip0.x() > ip1.x()) {
  //  swap(ip0, ip1);
  //}
  auto dx  = std::abs<long long>(ip1.x() - ip0.x());
  auto sx  = ip0.x() < ip1.x() ? 1 : -1;
  auto dy  = -std::abs<long long>(ip1.y() - ip0.y());
  auto sy  = ip0.y() < ip1.y() ? 1 : -1;
  auto err = dx + dy;
  auto e2  = (long long)(0);  // error value e_xy
  while (ip0.x() != ip1.x() || ip0.y() != ip1.y()) {
    pixels.push_back(ip0);
    e2 = 2 * err;
    if (e2 > dy) {
      err += dy;
      ip0.x() += sx;
    }  // e_xy+e_x > 0
    if (e2 < dx) {
      err += dx;
      ip0.y() += sy;
    }  // e_xy+e_y < 0
  }
  pixels.push_back(ip1);
  return pixels;
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
