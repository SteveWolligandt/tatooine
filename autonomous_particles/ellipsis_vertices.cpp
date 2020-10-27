#include "ellipsis_vertices.h"
//==============================================================================
auto ellipsis_vertices(mat<double, 2, 2> const& S, vec<double, 2> const& x0,
                       size_t const resolution) -> std::vector<vec<double, 3>> {
  std::vector<vec<double, 3>> ellipse;
  linspace                    radians{0.0, M_PI * 2, resolution + 1};
  radians.pop_back();
  for (auto t : radians) {
    auto const x = S * vec{std::cos(t), std::sin(t)} + x0;
    ellipse.emplace_back(x(0), x(1), 0);
  }
  return ellipse;
}
