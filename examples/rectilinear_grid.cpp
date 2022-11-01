#include <tatooine/rectilinear_grid.h>
using namespace tatooine;
auto main() -> int {
  auto g = rectilinear_grid{std::vector{0.0, 0.1, 1.0},
                            std::vector{0.0, 0.3, 0.6, 1.0}};
  auto& int_prop = g.vertex_property<int>("integer");
  auto& float_prop = g.vertex_property<float>("float");
  auto& double_prop = g.vertex_property<double>("double");
  int_prop(1, 1) = 3;
  float_prop(1, 1) = 3;
  double_prop(1, 1) = 3;
  g.write("rectilinear_grid.vtr");
}
