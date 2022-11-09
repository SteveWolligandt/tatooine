#include <tatooine/unstructured_tetrahedral_grid.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto grid = unstructured_tetrahedral_grid3{};
  auto v0   = grid.insert_vertex(0, 0, 0);
  auto v1   = grid.insert_vertex(1, 0, 0);
  auto v2   = grid.insert_vertex(0, 1, 0);
  auto v3   = grid.insert_vertex(1, 1, 0);
  auto v4   = grid.insert_vertex(0, 0, 1);
  auto v5   = grid.insert_vertex(1, 0, 1);
  auto v6   = grid.insert_vertex(0, 1, 1);
  auto v7   = grid.insert_vertex(1, 1, 1);

  auto& prop = grid.scalar_vertex_property("prop");
  prop[v0]   = 1;
  prop[v1]   = 2;
  prop[v2]   = 3;
  prop[v3]   = 4;
  prop[v4]   = 1;
  prop[v5]   = 2;
  prop[v6]   = 3;
  prop[v7]   = 4;

  grid.build_delaunay_mesh();

  for (auto s : grid.simplices()) {
    auto const& [v0, v1, v2, v3] = grid[s];
    std::cerr << v0.index() << ','
              << v1.index() << ','
              << v2.index() << ','
              << v3.index() << ",\n";
  }
  grid.write("example_unstructured_tetrahedral_grid.vtu");
}
