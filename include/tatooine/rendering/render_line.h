#ifndef TATOOINE_RENDERING_RENDER_LINE_H
#define TATOOINE_RENDERING_RENDER_LINE_H
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename Real, typename Callback>
auto render_line(Vec2<Real> p0, Vec2<Real> p1, int const line_width,
                 UniformRectilinearGrid2<Real> const& grid,
                 Callback&&                           callback) {
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
    return;
  }

  auto get_t = [](auto const a, auto const b, auto const c) {
    return (c - a) / (b - a);
  };
  // clamp p0.x()
  if (p0.x() < nn_ax_x.front()) {
    auto const t = get_t(p0.x(), p1.x(), nn_ax_x.front());
    p0           = vec{nn_ax_x.front(), p0.y() * (1 - t) + p1.y() * t};
  } else if (p0.x() > nn_ax_x.back()) {
    auto const t = get_t(p0.x(), p1.x(), nn_ax_x.back());
    p0           = vec{nn_ax_x.back(), p0.y() * (1 - t) + p1.y() * t};
  }
  // clamp p1.x()
  if (p1.x() <= nn_ax_x.front()) {
    auto const t = get_t(p0.x(), p1.x(), nn_ax_x.front());
    p1           = vec{nn_ax_x.front(), p0.y() * (1 - t) + p1.y() * t};
  } else if (p1.x() > nn_ax_x.back()) {
    auto const t = get_t(p0.x(), p1.x(), nn_ax_x.back());
    p1           = vec{nn_ax_x.back(), p0.y() * (1 - t) + p1.y() * t};
  }
  // clamp p0.y()
  if (p0.y() < nn_ax_y.front()) {
    auto const t = get_t(p0.y(), p1.y(), nn_ax_y.front());
    p0           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.front()};
  } else if (p0.y() > nn_ax_y.back()) {
    auto const t = get_t(p0.y(), p1.y(), nn_ax_y.back());
    p0           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.back()};
  }
  // clamp p1.y()
  if (p1.y() < nn_ax_y.front()) {
    auto const t = get_t(p0.y(), p1.y(), nn_ax_y.front());
    p1           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.front()};
  } else if (p1.y() > nn_ax_y.back()) {
    auto const t = get_t(p0.y(), p1.y(), nn_ax_y.back());
    p1           = vec{p0.x() * (1 - t) + p1.x() * t, nn_ax_y.back()};
  }
  auto ip0 = Vec2<std::size_t>{};
  auto ip1 = Vec2<std::size_t>{};
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
  auto const dist =
      vec{std::max(ip0.x(), ip1.x()) - std::min(ip0.x(), ip1.x()),
          std::max(ip0.y(), ip1.y()) - std::min(ip0.y(), ip1.y())};

  auto dx  = std::abs((long long)(ip1.x()) - (long long)(ip0.x()));
  auto sx  = ip0.x() < ip1.x() ? 1 : -1;
  auto dy  = -std::abs((long long)(ip1.y()) - (long long)(ip0.y()));
  auto sy  = ip0.y() < ip1.y() ? 1 : -1;
  auto err = dx + dy;
  auto e2  = (long long)(0);  // error value e_xy
  auto c   = [&](auto const t, auto const ix, auto const iy) {
    if (line_width == 1) {
      callback(t, ix, iy);
    } else {
      auto const half_line_width = line_width / 2;
      auto const m               = 1 - line_width % 2;
      if (dist.x() < dist.y()) {
        for (int i = -half_line_width + m; i <= half_line_width; ++i) {
          if (ix >= std::abs(i) &&
              std::size_t(ix + i) < grid.template size<0>() - 1) {
            callback(t, ix + i, iy);
          }
        }
      } else {
        for (int i = -half_line_width + m; i <= half_line_width; ++i) {
          if (iy >= std::abs(i) &&
              std::size_t(iy + i) < grid.template size<1>() - 1) {
            callback(t, ix, iy + i);
          }
        }
      }
    }
  };

  auto const t_offset = Real(1) / (std::max(dist.x(), dist.y()) - 1);
  auto i = std::size_t(0);
  while (ip0.x() != ip1.x() || ip0.y() != ip1.y()) {
    c(i++ * t_offset, ip0.x(), ip0.y());
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
  c(i++ * t_offset, ip1.x(), ip1.y());
}
//------------------------------------------------------------------------------
template <typename Real, typename Callback>
auto render_line(Vec2<Real> const& p0, Vec2<Real> const& p1,
                 UniformRectilinearGrid2<Real> const& grid,
                 Callback&&                           callback) {
  render_line(p0, p1, 1, grid, std::forward<Callback>(callback));
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
