#include <tatooine/ridgelines.h>
#include <tatooine/isolines.h>
using namespace tatooine;
auto main() -> int {
  auto grid = rectilinear_grid{linspace{-1.0, 1.0, 100}, linspace{-1.0, 1.0, 100}};
  auto const& f =
      grid.sample_to_vertex_property([](auto const& x) { return gcem::sin(x.x()); }, "f");
  auto g_tmp = diff(f);
  auto H_tmp = diff(g_tmp);
  auto & g = grid.sample_to_vertex_property(
      [&](integral auto const... is) { return g_tmp(is...); }, "g");
  auto & H = grid.sample_to_vertex_property(
      [&](integral auto const... is) { return H_tmp(is...); }, "H");
  auto & Hg = grid.sample_to_vertex_property(
      [&](integral auto const... is) { return H(is...) * g(is...); }, "Hg");
  auto& det = grid.sample_to_vertex_property(
      [&](integral auto const... is) {
        return g(is...)(0) * Hg(is...)(1) - g(is...)(1) * Hg(is...)(0);
      },
      "det(g|Hg)");
  auto const ridges = isolines(det, 0);
  write(std::vector{ridges}, "ridges.vtp");
  grid.write("ridge_data.vtr");
}
