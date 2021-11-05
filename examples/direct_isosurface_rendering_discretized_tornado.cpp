#include <tatooine/Q_field.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/rendering/direct_isosurface.h>
#include <tatooine/rendering/perspective_camera.h>
//==============================================================================
namespace tatooine::examples {
//==============================================================================
auto direct_iso_discretized_tornado() {
  auto const v = analytical::fields::numerical::tornado{};
  auto       discretized_domain =
      grid{linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501},
           linspace{-1.0, 1.0, 501}};
  auto&        discretized_velocity = discretize(v, discretized_domain, "v", 0);
  auto         discretized_jacobian = diff(discretized_velocity);
  auto&        discretized_Q = discretized_domain.scalar_vertex_property("Q");
  real_t const t             = 0;
  discretized_domain.vertices().iterate_indices(
      [&](auto const... is) {
        mat3 const J         = discretized_jacobian(is...);
        mat3 const S         = (J + transposed(J)) / 2;
        mat3 const Omega     = (J - transposed(J)) / 2;
        discretized_Q(is...) = (sqr_norm(Omega, 2) - sqr_norm(S, 2)) / 2;
      },
      execution_policy::parallel);
  color_scales::viridis         color_scale;
  size_t const                  width = 2000, height = 2000;
  auto const                    eye    = vec3{2, 2, 2};
  auto const                    lookat = vec3{0, 0, 0};
  auto const                    up     = vec3{0, 0, 1};
  auto const                    fov    = 60;
  rendering::perspective_camera cam{eye, lookat, up, fov, width, height};
  real_t const                  min      = 0.0;
  real_t const                  max      = 1.0;
  real_t const                  isovalue = 0.1;

  auto Q_sampler = discretized_Q.linear_sampler();
  auto shader    = [&](auto const& x_iso, auto const& gradient,
                    auto const& view_dir) {
    auto const normal      = normalize(gradient);
    auto const diffuse     = std::abs(dot(view_dir, normal));
    auto const reflect_dir = reflect(-view_dir, normal);
    auto const spec_dot = std::max(std::abs(dot(reflect_dir, view_dir)), 0.0);
    auto const specular = std::pow(spec_dot, 100);
    auto const scalar =
        std::clamp<real_t>((length(v)(x_iso, t) - min) / (max - min), 0, 1);
    auto const albedo = color_scale(scalar);
    auto const col    = albedo * diffuse + specular;
    return vec{col(0), col(1), col(2), scalar * scalar};
  };
  auto const rendering_grid =
      rendering::direct_isosurface(cam, Q_sampler, isovalue, shader);
#ifdef TATOOINE_PNG_AVAILABLE
  write_png("direct_iso_discretized_tornado_Q.png",
            rendering_grid.vec3_vertex_property("rendered_isosurface"));
#endif
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main() -> int {
  using namespace tatooine::examples;
  direct_iso_discretized_tornado();
}
