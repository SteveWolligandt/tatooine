#include <tatooine/unstructured_triangular_grid.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto grid = unstructured_triangular_grid2{};
  auto v0 = grid.insert_vertex(0, 0);
  auto v1 = grid.insert_vertex(1, 0);
  auto v2 = grid.insert_vertex(1, 1);
  auto v3 = grid.insert_vertex(0, 1);
  auto v4 = grid.insert_vertex(2, 0.5);

  auto& prop = grid.scalar_vertex_property("prop");
  prop[v0]   = 1;
  prop[v1]   = 2;
  prop[v2]   = 3;
  prop[v3]   = 4;
  prop[v4]   = 5;

  grid.write("example_pointset.vtk");
  grid.write("example_pointset.vtp");
}
