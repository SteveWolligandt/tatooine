#include <tatooine/rectilinear_grid.h>
using namespace tatooine;
auto main() -> int {
  auto lores_grid =
      rectilinear_grid{linspace{-1.0, 1.0, 10}, linspace{-1.0, 1.0, 10}};
  auto constexpr f = [](auto&& x) { return sin(x.x() * 10) * cos(x.y() * 10); };
  auto sampler = lores_grid
                     .sample_to_vertex_property(f, "prop",
                                                execution_policy::parallel)
                     .cubic_sampler();
  auto hires_grid =
      rectilinear_grid{linspace{-1.0, 1.0, 1000}, linspace{-1.0, 1.0, 1000}};
  hires_grid.sample_to_vertex_property(sampler, "prop");

  lores_grid.write("lores.vtr");
  hires_grid.write("hires.vtr");
  sampler(0,0);
}
