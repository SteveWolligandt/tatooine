#include <tatooine/Q_field.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/direct_iso.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto direct_iso_tornado() {
  analytical::fields::numerical::tornado v;
  color_scales::viridis                  color_scale;
  axis_aligned_bounding_box              aabb{vec{-10.0, -10.0, -10.0},
                                 vec{10.0, 10.0, 10.0}};
  size_t const                           width = 500, height = 250;
  auto const                             eye                 = vec3{20, 20, 20};
  auto const                             lookat              = vec3::zeros();
  auto const                             up                  = vec3{0, 0, 1};
  auto const                             fov                 = 60;
  auto const                             near_plane_distance = 0.001;
  auto const                             far_plane_distance  = 1000;
  rendering::perspective_camera<double>  cam{
      eye,   lookat, up, fov, near_plane_distance, far_plane_distance,
      width, height};
  constexpr auto alpha            = [](auto const t) -> double { return 0.5; };
  auto const     t                = 0;
  auto const     min              = 0;
  auto const     max              = 100;
  auto const     distance_on_ray  = 0.001;
  auto const     isovalue         = 1;
  auto const     background_color = vec3::ones();
  auto const     Q                = [&](auto const& x) -> decltype(auto) {
    return tatooine::Q(v)(x, t);
  };
  auto const domain_check = [&aabb](auto const& x) {
    return aabb.is_inside(x);
  };
  auto const Q_gradient = [&](auto const& x) -> decltype(auto) {
    constexpr auto eps      = 1e-5;
    auto gradient = vec3{};
    auto offset = vec3::zeros();
    for (size_t i = 0; i < 3; ++i) {
      offset(i) = eps;
      auto fw = x + offset;
      auto bw = x - offset;
      auto dx   = 2 * eps;
      if (!domain_check(fw)) {
        fw = x;
        dx = eps;
      }
      if (!domain_check(bw)) {
        bw = x;
        dx = eps;
      }
      offset(i) = 0;
      gradient(i) = (Q(fw) - Q(bw)) / dx;
    }
    return gradient;
  };
  auto const mag = [&](auto const& x) -> decltype(auto) {
    return length(v)(x, t);
  };
  auto const rendering_grid = direct_iso(
      cam, aabb, Q, Q_gradient, isovalue, mag, min, max,
      domain_check, distance_on_ray,
      color_scale, alpha, background_color);
#ifdef TATOOINE_HAS_PNG_SUPPORT
  write_png("direct_iso_analytical_tornado_Q.png",
            rendering_grid.vertex_property<vec3>("rendering"));
#endif
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main(int argc, char const** argv) -> int {
  using namespace tatooine::examples;
  direct_iso_tornado();
}
