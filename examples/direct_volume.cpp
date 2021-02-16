#include <tatooine/Q_field.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/direct_volume_rendering.h>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto direct_volume_rendering_tornado() {
  analytical::fields::numerical::tornado v;
  color_scales::viridis                  color_scale;
  axis_aligned_bounding_box              aabb{vec{-10.0, -10.0, -10.0},
                                              vec{ 10.0,  10.0,  10.0}};
  size_t const                           width = 1000, height = 500;
  auto const                             eye                 = vec3{20, 20, 20};
  auto const                             lookat              = vec3::zeros();
  auto const                             up                  = vec3{0, 0, 1};
  auto const                             fov                 = 60;
  auto const                             near_plane_distance = 0.001;
  auto const                             far_plane_distance  = 1000;
  rendering::perspective_camera<double>  cam{
      eye,   lookat, up, fov, near_plane_distance, far_plane_distance,
      width, height};
  constexpr auto alpha = [](auto const t) -> double {
    auto const min    = 0;
    auto const max    = 0.01;
    if (t < 0) {
      return min;
    } else if (t > 1) {
      return max;
    } else {
      return t * t * (max - min) + min;
    }
  };
  auto const t                = 0;
  auto const min              = 0.001;
  auto const max              = 0.01;
  auto const distance_on_ray  = 0.001;
  auto const background_color = vec3::ones();
  auto       Q_grid =
      direct_volume_rendering(cam, aabb, Q(v), t, min, max, distance_on_ray,
                              color_scale, alpha, background_color);
  write_png("direct_volume_tornado_Q.png",
            Q_grid.vertex_property<vec<double, 3>>("rendering"));
  Q_grid.write_vtk("direct_volume_tornado_Q.vtk");
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main(int argc, char const** argv) -> int {
  using namespace tatooine::examples;
  direct_volume_rendering_tornado();
}
