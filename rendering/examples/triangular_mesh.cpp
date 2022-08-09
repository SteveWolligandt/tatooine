#include <tatooine/rendering/interactive.h>
using namespace tatooine;
auto main() -> int {
  auto mesh = unstructured_triangular_grid3{};
  auto v0 = mesh.insert_vertex(0,0,0);
  auto v1 = mesh.insert_vertex(1,0,0);
  auto v2 = mesh.insert_vertex(0,1,0);
  auto v3 = mesh.insert_vertex(1,1,0);
  mesh.insert_triangle(v0, v1, v3);
  mesh.insert_triangle(v0, v3, v2);
  mesh.scalar_vertex_property("foo");
  rendering::interactive::show(mesh);
}
