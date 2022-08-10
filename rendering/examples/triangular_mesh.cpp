#include <tatooine/rendering/interactive.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/random.h>
using namespace tatooine;
auto main() -> int {

  auto mesh = geometry::discretize(geometry::sphere3{1.0}, 4);
  auto & foo = mesh.scalar_vertex_property("foo");
  auto rand = random::uniform{0.0, 1.0};
  for (auto const v : mesh.vertices()) {
    foo[v] = rand();
  }
  rendering::interactive::show(mesh);
}
