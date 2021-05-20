#include <tatooine/Q_field.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/direct_iso.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto direct_iso_discretized_tornado() {
  analytical::fields::numerical::tornado v;
  auto                                   discretized_domain =
      grid{linspace{-1.0, 1.0, 51},
           linspace{-1.0, 1.0, 51},
           linspace{-1.0, 1.0, 51}};
  auto& discretized_field = discretize(v, discretized_domain, "v", 0);
  auto& discretized_Q     = discretized_domain.add_scalar_vertex_property("Q");
  auto  discretized_Q_sampler      = discretized_Q.cubic_sampler();
  auto  diff_discretized_Q_sampler = diff(discretized_Q_sampler);
  auto  discretized_J              = diff(discretized_field);

  discretized_domain.parallel_loop_over_vertex_indices([&](auto const... is) {
    auto const J = discretized_J(is...);
    discretized_Q(is...) =
        0.5 * ((J(0, 0) + J(1, 1) + J(2, 2)) * (J(0, 0) + J(1, 1) + J(2, 2)) -
               (J(0, 0) * J(0, 0) + J(1, 1) * J(1, 1) + J(2, 2) * J(2, 2))) -
        J(0, 1) * J(1, 0) - J(0, 2) * J(2, 0) - J(1, 2) * J(2, 1);
  });

  color_scales::viridis color_scale;
  auto const            aabb  = discretized_domain.bounding_box();
  size_t const          width = 500, height = 250;
  auto const            eye                 = vec3{2, 2, 2};
  auto const            lookat              = vec3::zeros();
  auto const            up                  = vec3{0, 0, 1};
  auto const            fov                 = 60;
  rendering::perspective_camera cam{eye, lookat, up, fov, width, height};
  constexpr auto alpha        = [](auto const t) -> real_t { return 1; };
  auto const     t            = 0;
  auto const     min          = 0;
  auto const     max          = 1;
  auto const distance_on_ray  = discretized_domain.dimension<0>().spacing() / 10;
  auto const isovalue         = 0.1;
  auto const background_color = vec3::ones();
  auto const Q                = [&](auto const& x) -> real_t {
    return discretized_Q_sampler(x);
  };
  auto const Q_gradient = [&](auto const& x) -> vec3 {
    return diff_discretized_Q_sampler(x);
  };
  auto const mag = [&](auto const& x) -> decltype(auto) {
    return length(v)(x, t);
  };
  auto const rendering_grid = direct_iso(
      cam, aabb, Q, Q_gradient, isovalue, mag, min, max,
      [&aabb](auto const& x) { return aabb.is_inside(x); }, distance_on_ray,
      color_scale, alpha, background_color);
#ifdef TATOOINE_HAS_PNG_SUPPORT
  write_png("direct_iso_discretized_tornado_Q.png",
            rendering_grid.vertex_property<vec3>("rendering"));
#endif
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main(int argc, char const** argv) -> int {
  using namespace tatooine::examples;
  direct_iso_discretized_tornado();
}
