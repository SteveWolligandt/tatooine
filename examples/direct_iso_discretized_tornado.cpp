#include <tatooine/Q_field.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/direct_isosurface_rendering.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto direct_iso_discretized_tornado() {
  auto const v = analytical::fields::numerical::tornado{};
  auto                                   discretized_domain =
      grid{linspace{-1.0, 1.0, 501},
           linspace{-1.0, 1.0, 501},
           linspace{-1.0, 1.0, 501}};
  auto& discretized_field = discretize(v, discretized_domain, "v", 0);
  auto  discretized_J              = diff(discretized_field);


  color_scales::viridis color_scale;
  auto const            aabb  = discretized_domain.bounding_box();
  size_t const          width = 2000, height = 2000;
  auto const            eye                 = vec3{2, 2, 2};
  auto const            lookat              = vec3::zeros();
  auto const            up                  = vec3{0, 0, 1};
  auto const            fov                 = 60;
  rendering::perspective_camera cam{eye, lookat, up, fov, width, height};
  real_t const                  t              = 0;
  real_t const                  min            = 0.0;
  real_t const                  max            = 1.0;
  real_t const                  isovalue       = 0.1;

  auto& discretized_Q = discretized_domain.add_scalar_vertex_property("Q");
  discretized_domain.parallel_loop_over_vertex_indices([&](auto const... is) {
    //discretized_Q(is...) = Q(v)(discretized_domain(is...), t);
    auto const J = discretized_J(is...);
    discretized_Q(is...) =
        0.5 * ((J(0, 0) + J(1, 1) + J(2, 2)) * (J(0, 0) + J(1, 1) + J(2, 2)) -
               (J(0, 0) * J(0, 0) + J(1, 1) * J(1, 1) + J(2, 2) * J(2, 2))) -
        J(0, 1) * J(1, 0) - J(0, 2) * J(2, 0) - J(1, 2) * J(2, 1);
  });
  auto Q_sampler = discretized_Q.linear_sampler();

  auto const                    rendering_grid = direct_isosurface_rendering(
      cam, Q_sampler, isovalue,
      [&](auto const& x_iso, auto const& gradient, auto const& view_dir) {
        auto const normal      = normalize(gradient);
        auto const diffuse     = std::abs(dot(view_dir, normal));
        auto const reflect_dir = reflect(-view_dir, normal);
        auto const spec_dot =
            std::max(std::abs(dot(reflect_dir, view_dir)), 0.0);
        auto const specular = std::pow(spec_dot, 100);
        auto const albedo   = color_scale(
            std::clamp<real_t>((length(v)(x_iso, t) - min) / (max - min), 0, 1));
        auto const col      = albedo * diffuse + specular;
        return vec{col(0), col(1), col(2), 0.7};
      });
#ifdef TATOOINE_HAS_PNG_SUPPORT
  write_png("direct_iso_discretized_tornado_Q.png",
            rendering_grid.vertex_property<vec3>("rendered_isosurface"));
#endif
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main(int argc, char const** argv) -> int {
  using namespace tatooine::examples;
  direct_iso_discretized_tornado();
}
