#include <tatooine/rendering/render_line.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto grid =
      uniform_rectilinear_grid2{linspace{0.0, 1.0, 11}, linspace{0.0, 1.0, 11}};
  auto& rasterized_line = grid.vec3_vertex_property("line");
  grid.vertices().iterate_indices(
      [&](auto const... is) { rasterized_line(is...) = vec3::ones(); },
      execution_policy::parallel);
  auto pixels = rendering::render_line(vec2{0.2, 0.2}, vec2{0.8, 0.8}, grid);
  for (auto const& ix : pixels) {
    rasterized_line(ix(0), ix(1)) = vec3::zeros();
  }
  rasterized_line.write_png("rasterized_line.png");
}
