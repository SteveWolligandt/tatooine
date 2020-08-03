#include <tatooine/direct_volume_rendering.h>
#include <tatooine/perspective_camera.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("direct_volume_rendering_abc_magnitude",
          "[direct_volume_rendering][abc][magnitude]") {
  analytical::fields::numerical::abcflow v;
  grid<linspace<double>, linspace<double>, linspace<double>> g{
      linspace{-1.0, 1.0, 20}, linspace{-1.0, 1.0, 20}, linspace{-1.0, 1.0, 20}};
  auto& mag = g.add_contiguous_vertex_property<double>("mag");
  double min = std::numeric_limits<double>::max(), max = -std::numeric_limits<double>::max();
  g.loop_over_vertex_indices([&](auto const... is) {
    mag.container().at(is...) = length(v(g.vertex_at(is...), 0));
    min = std::min(mag.container().at(is...), min);
    max = std::max(mag.container().at(is...), max);
  });
  perspective_camera<double> cam{vec{2, 2, 2}, vec{0, 0, 0}, 60, 500, 500};
  auto rendered_grid = direct_volume_rendering(cam, mag, 1, max, 0.5);
  rendered_grid.write_vtk("direct_volume_abc_mag.vtk");
}
//==============================================================================
}
//==============================================================================
