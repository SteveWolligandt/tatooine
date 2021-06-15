#include <tatooine/Q_field.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/color_scales/magma.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/rendering/direct_volume.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto direct_volume_rendering_tornado() {
  analytical::fields::numerical::tornado v;
  color_scales::viridis                  color_scale;
  axis_aligned_bounding_box     aabb{vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}};
  size_t const                  width = 2000, height = 2000;
  auto const                    eye    = vec3{2, 2, 2};
  auto const                    lookat = vec3::zeros();
  auto const                    up     = vec3{0, 0, 1};
  auto const                    fov    = 60;
  rendering::perspective_camera cam{eye, lookat, up, fov, width, height};
  constexpr auto                alpha = [](auto const t) -> double {
    auto const min = 0;
    auto const max = 0.1;
    if (t < 0) {
      return min;
    } else if (t > 1) {
      return max;
    } else {
      return t * t * (max - min) + min;
    }
  };
  auto const t               = 0;
  auto const min             = 0;
  auto const max             = 1;
  auto const distance_on_ray = 0.001;
  auto const qv              = Q(v);
  auto       rendering_grid  = rendering::direct_volume(
      cam, aabb, [](auto const&) { return true; }, distance_on_ray,
      [&](auto const& x, auto const&) {
        auto const t   = (qv(x) - min) / (max - min);
        auto const rgb = color_scale(t);
        return vec4{rgb(0), rgb(1), rgb(2), alpha(t)};
      });
#ifdef TATOOINE_HAS_PNG_SUPPORT
  write_png("direct_volume_tornado_Q.png",
            rendering_grid.vertex_property<vec<double, 3>>("rendering"));
#endif
  rendering_grid.write_vtk("direct_volume_tornado_Q.vtk");
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main() -> int {
  using namespace tatooine::examples;
  direct_volume_rendering_tornado();
}
