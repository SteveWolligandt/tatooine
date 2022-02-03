#include <tatooine/rendering/interactive.h>
using namespace tatooine;
auto main() -> int {
  auto  g    = rectilinear_grid{linspace{0.0, 1.0, 20}, linspace{0.0, 1.0, 20}};
  auto& p    = g.scalar_vertex_property("test");
  auto& p2    = g.scalar_vertex_property("test2");
  auto& p3    = g.vec2_vertex_property("vec");
  auto  rand  = random::uniform<real_t>{};
  g.vertices().iterate_indices([&](auto const... is) {
    p(is...)  = rand();
    p2(is...) = rand();
    p3(is...) = vec{rand(), rand()};
  });
  rendering::interactive(g, geometry::ellipse{vec2f{0, 0}, 0.5f, 1.0f},
                         geometry::ellipse{vec2f{1, 0}, 0.5f, 1.0f},
                         geometry::ellipse{vec2f{0, 2}, 0.5f, 1.0f},
                         geometry::ellipse{vec2f{1, 2}, 0.5f, 1.0f});
}
