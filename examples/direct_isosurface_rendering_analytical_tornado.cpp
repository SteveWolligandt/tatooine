#include <tatooine/Q_field.h>
#include <tatooine/steady_field.h>
#include <tatooine/differentiated_field.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/rendering/direct_isosurface.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto direct_iso_tornado() {
  analytical::fields::numerical::tornado v;
  color_scales::viridis                  color_scale;
  axis_aligned_bounding_box aabb{vec{-1.0, -1.0, -1.0}, vec{1.0, 1.0, 1.0}};
  size_t const          width = 2000, height = 2000;
  auto const            eye                 = vec3{2, 2, 2};
  auto const            lookat              = vec3::zeros();
  auto const            up                  = vec3{0, 0, 1};
  auto const            fov                 = 60;
  rendering::perspective_camera cam{eye, lookat, up, fov, width, height};
  auto const                    t                = 0;
  real_t const                  min              = 0.0;
  real_t const                  max              = 1.0;
  auto const                    distance_on_ray  = 0.001;
  auto const                    isovalue         = 1;
  auto const                    domain_check     = [&aabb](auto const& x) {
    return aabb.is_inside(x);
  };
  auto shader =
      [&](auto const& x_iso, auto const& view_dir) {
        auto const normal      = normalize(diff(Q(v), 1e-7)(x_iso, t));
        auto const diffuse     = std::abs(dot(view_dir, normal));
        auto const reflect_dir = reflect(-view_dir, normal);
        auto const spec_dot =
            std::max(std::abs(dot(reflect_dir, view_dir)), 0.0);
        auto const specular = std::pow(spec_dot, 100);
        auto const albedo   = color_scale(
            std::clamp<real_t>((length(v)(x_iso, t) - min) / (max - min), 0, 1));
        auto const col      = albedo * diffuse + specular;
        return vec{col(0), col(1), col(2), 1};
      };
  auto const rendering_grid =
      rendering::direct_isosurface(cam, aabb, steady(Q(v), t), domain_check,
                                  isovalue, distance_on_ray, shader);
#ifdef TATOOINE_HAS_PNG_SUPPORT
  write_png("direct_isosurface_rendering_analytical_tornado_Q_with_velocity_magnitude.png",
            rendering_grid.vertex_property<vec3>("rendered_isosurface"));
#endif
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main() -> int {
  using namespace tatooine::examples;
  direct_iso_tornado();
}
