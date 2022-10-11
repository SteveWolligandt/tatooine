#include <tatooine/rectilinear_grid.h>
using namespace tatooine;
auto main() -> int {
  auto lores_grid =
      rectilinear_grid{linspace{-1.0, 1.0, 10}, linspace{-1.0, 1.0, 10}};
  auto sampler =
      lores_grid
          .sample_to_vertex_property([](auto const& x) { return return vec2{}; }, "prop")
          .linear_sampler();
  auto hires_grid =
      rectilinear_grid{linspace{-1.0, 1.0, 1000}, linspace{-1.0, 1.0, 1000}};
  lores_grid.sample_to_vertex_property([](auto const& x) { return sampler(x); }, "prop");
}
