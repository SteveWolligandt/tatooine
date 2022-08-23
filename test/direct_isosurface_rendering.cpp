#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/field_operations.h>
#include <tatooine/line.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/direct_isosurface.h>
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/spacetime_vectorfield.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("direct_iso_grid_vertex_sampler") {
  auto v       = analytical::numerical::doublegyre{};
  auto vst     = spacetime_vectorfield{v};
  auto mag_vst = euclidean_length(vst);

  auto  g    = rectilinear_grid{linspace{0.0, 2.0, 20}, linspace{0.0, 1.0, 10},
                            linspace{0.0, 10.0, 100}};
  auto  rand = random::uniform{-1.0, 1.0};
  auto& s    = g.sample_to_vertex_property(mag_vst, "s");
  auto const iso    = 1.05;
  auto const eye    = vec3{-6, 4, 5};
  auto const lookat = vec3{1, 0, 5};

  auto sampler = s.linear_sampler();
  auto cam =
      rendering::perspective_camera{eye, lookat, vec3{0, 1, 0}, 60, 10, 5};

  rendering::direct_isosurface(
      cam, sampler, iso,
      [](auto const& x_iso, auto const /*iso*/, auto const& gradient,
         auto const& view_dir, auto const& /*pixel_coord*/) {
        auto const normal      = normalize(gradient);
        auto const diffuse     = std::abs(dot(view_dir, normal));
        auto const reflect_dir = reflect(-view_dir, normal);
        auto const spec_dot =
            std::max(std::abs(dot(reflect_dir, view_dir)), 0.0);
        auto const specular = std::pow(spec_dot, 100);
        auto const albedo   = vec3{std::cos(x_iso(0) * 10) * 0.5 + 0.5,
                                 std::sin(x_iso(1) * 20) * 0.5 + 0.5, 0.1};
        auto const col      = albedo * diffuse + specular;
        return col;
      });
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
