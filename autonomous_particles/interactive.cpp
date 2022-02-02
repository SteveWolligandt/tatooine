#include <tatooine/rendering/interactive.h>
using namespace tatooine;
auto main() -> int {
  auto  g    = rectilinear_grid{linspace{0.0, 1.0, 20}, linspace{0.0, 1.0, 20}};
  auto& p    = g.scalar_vertex_property("test");
  auto  rand = random::uniform<real_t>{};
  g.vertices().iterate_indices([&](auto const... is) { p(is...) = rand(); });
  rendering::interactive(g, p, geometry::ellipse{vec2f{0, 0}, 0.5f, 1.0f},
                         geometry::ellipse{vec2f{1, 0}, 0.5f, 1.0f},
                         geometry::ellipse{vec2f{0, 2}, 0.5f, 1.0f},
                         geometry::ellipse{vec2f{1, 2}, 0.5f, 1.0f});
}
