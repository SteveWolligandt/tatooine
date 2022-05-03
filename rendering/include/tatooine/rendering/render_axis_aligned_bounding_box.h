#ifndef TATOOINE_RENDERING_RENDER_AXIS_ALIGNED_BOUNDING_BOX_H
#define TATOOINE_RENDERING_RENDER_AXIS_ALIGNED_BOUNDING_BOX_H
//==============================================================================
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/concepts.h>
#include <tatooine/rendering/render_line.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <typename Real>
auto render(
    AABB3<Real> const& aabb, int const line_width,
    UniformRectilinearGrid2<Real> const& grid, camera auto const& cam,
    invocable<Real, typename AABB3<Real>::pos_type, typename AABB3<Real>::pos_type,
              std::size_t, std::size_t> auto&& callback) {
  auto r = [&](auto const x0, auto const x1) {
    render(x0, x1, line_width, grid, cam,
           [&](auto const t, auto const... is) { callback(t, x0, x1, is...); });
  };
  r(Vec4<Real>{aabb.min(0), aabb.min(1), aabb.min(2), 1},
    Vec4<Real>{aabb.max(0), aabb.min(1), aabb.min(2), 1});
  r(Vec4<Real>{aabb.min(0), aabb.max(1), aabb.min(2), 1},
    Vec4<Real>{aabb.max(0), aabb.max(1), aabb.min(2), 1});
  r(Vec4<Real>{aabb.min(0), aabb.min(1), aabb.max(2), 1},
    Vec4<Real>{aabb.max(0), aabb.min(1), aabb.max(2), 1});
  r(Vec4<Real>{aabb.min(0), aabb.max(1), aabb.max(2), 1},
    Vec4<Real>{aabb.max(0), aabb.max(1), aabb.max(2), 1});

  r(Vec4<Real>{aabb.min(0), aabb.min(1), aabb.min(2), 1},
    Vec4<Real>{aabb.min(0), aabb.max(1), aabb.min(2), 1});
  r(Vec4<Real>{aabb.max(0), aabb.min(1), aabb.min(2), 1},
    Vec4<Real>{aabb.max(0), aabb.max(1), aabb.min(2), 1});
  r(Vec4<Real>{aabb.min(0), aabb.min(1), aabb.max(2), 1},
    Vec4<Real>{aabb.min(0), aabb.max(1), aabb.max(2), 1});
  r(Vec4<Real>{aabb.max(0), aabb.min(1), aabb.max(2), 1},
    Vec4<Real>{aabb.max(0), aabb.max(1), aabb.max(2), 1});

  r(Vec4<Real>{aabb.min(0), aabb.min(1), aabb.min(2), 1},
    Vec4<Real>{aabb.min(0), aabb.min(1), aabb.max(2), 1});
  r(Vec4<Real>{aabb.max(0), aabb.min(1), aabb.min(2), 1},
    Vec4<Real>{aabb.max(0), aabb.min(1), aabb.max(2), 1});
  r(Vec4<Real>{aabb.min(0), aabb.max(1), aabb.min(2), 1},
    Vec4<Real>{aabb.min(0), aabb.max(1), aabb.max(2), 1});
  r(Vec4<Real>{aabb.max(0), aabb.max(1), aabb.min(2), 1},
    Vec4<Real>{aabb.max(0), aabb.max(1), aabb.max(2), 1});
}
//------------------------------------------------------------------------------
template <typename Real, typename Callback>
auto render(
    AABB3<Real> const& aabb, UniformRectilinearGrid2<Real> const& grid,
    camera auto const&                         cam,
    invocable<Real, typename AABB3<Real>::pos_type, typename AABB3<Real>::pos_type,
              std::size_t, std::size_t> auto&& callback) {
  render(aabb, 1, grid, cam, std::forward<Callback>(callback));
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
