#include <tatooine/edgeset.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto grid = edgeset2{};
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

  grid.insert_cell(v0, v1);
  grid.insert_cell(v1, v4);
  grid.insert_cell(v4, v2);
  grid.insert_cell(v2, v3);
  grid.insert_cell(v3, v0);

  for (auto c : grid.cells()) {
    auto const& [v0, v1] = grid[c];
    std::cerr << v0.index() << '\n';
    std::cerr << v1.index() << "\n\n";
  }
  grid.write("example_edgeset.vtk");
  grid.write("example_edgeset.vtp");
}
