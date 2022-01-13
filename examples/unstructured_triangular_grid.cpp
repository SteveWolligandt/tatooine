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

  grid.insert_cell(v0, v1, v2);
  grid.insert_cell(v0, v2, v3);
  grid.insert_cell(v1, v4, v2);

  for (auto c : grid.cells()) {
    auto const& [v0, v1, v2] = grid[c];
    std::cerr << v0.index() << '\n';
    std::cerr << v1.index() << '\n';
    std::cerr << v2.index() << "\n\n";
  }
  grid.write("example_unstructured_triangular_grid.vtk");
  grid.write("example_unstructured_triangular_grid.vtp");
}
