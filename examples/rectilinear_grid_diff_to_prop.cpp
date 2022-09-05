#include <tatooine/rectilinear_grid.h>
using namespace tatooine;
auto main() -> int {
  auto const scalar_field = [](auto const& p) {
    return p.x() * p.x() * p.y() * p.y();
  };
  auto grid =
      rectilinear_grid{linspace{-1.0, 1.0, 1000}, linspace{1.0, 3.0, 110}};
  auto const& discretized_scalar_field = grid.sample_to_vertex_property(scalar_field, "scalar");
  auto        diff1_scalar = diff(discretized_scalar_field, 5);
  auto        diff2_scalar = diff(diff1_scalar);
  grid.sample_to_vertex_property(
      [&](integral auto const... is) { return diff1_scalar(is...); },
      "first_derivative");
  grid.sample_to_vertex_property(
      [&](integral auto const... is) { return diff2_scalar(is...); },
      "second_derivative");
  grid.write("rectilinear_grid.diff_to_prop.vtr");
}
