#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/field_operations.h>
#include <tatooine/line.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/direct_isosurface.h>
#include <tatooine/rendering/perspective_camera.h>
#include <tatooine/spacetime_vectorfield.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("direct_iso_grid_vertex_sampler") {
  auto v       = analytical::numerical::doublegyre{};
  auto vst     = spacetime_vectorfield{v};
  auto mag_vst = euclidean_length(vst);

  auto  g   = rectilinear_grid{
    linspace{0.0, 2.0, 200},
    linspace{0.0, 1.0, 100},
    linspace{0.0, 10.0, 1000}};
  auto rand = random::uniform{-1.0, 1.0};
  auto& s =
      g.sample_to_vertex_property(mag_vst, "s");
  auto  iso = 1.05;
  auto  eye = line3{};
  eye.push_back(-2, 4, -2);
  eye.parameterization().back() = 0;
  eye.push_back(-6, 4, 5);
  eye.parameterization().back() = 0.5;
  eye.push_back(-2, 4, 12);
  eye.parameterization().back() = 1;
  auto eye_track                = eye.cubic_sampler();
  auto lookat                   = line3{};
  lookat.push_back(1, 0, 3);
  lookat.parameterization().back() = 0;
  lookat.push_back(1, 0, 7);
  lookat.parameterization().back() = 1;
  auto lookat_track = lookat.linear_sampler();

  auto       sampler = s.linear_sampler();
  size_t     i       = 0;
  auto const ts      = linspace{0.0, 1.0, 100};
  for (auto const t : ts) {
    auto cam = rendering::perspective_camera{
        eye_track(t), lookat_track(t), vec3{0, 1, 0}, 60, 1000, 500};

    auto ss = std::stringstream {};
    ss << std::setfill('0') << std::setw(2) << i;
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
        })
        .vertex_property<vec3>("rendered_isosurface")
        .write_png("direct_isosurface_rendering." + ss.str() + ".png");
    ++i;
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
