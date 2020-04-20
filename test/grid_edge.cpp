#include <tatooine/grid.h>
#include <tatooine/vtk_legacy.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("grid_edge_max_num_edges_per_cell", "[grid][grid_edge]")  {
  REQUIRE(grid_edge_iterator<double, 2>::max_num_edges_per_cell == 4);
  REQUIRE(grid_edge_iterator<double, 3>::max_num_edges_per_cell == 13);
  REQUIRE(grid_edge_iterator<double, 4>::max_num_edges_per_cell == 40);
}
//==============================================================================
TEST_CASE("grid_edge_neighbor_dirs", "[grid][grid_edge]") {
  auto dirs2 = grid_edge_iterator<double, 2>::edge_dirs;
  auto bases2 = grid_edge_iterator<double, 2>::bases;
  for (size_t i = 0; i < grid_edge_iterator<double, 2>::max_num_edges_per_cell;
       ++i) {
    std::cerr << dirs2[i](0) << ", " << dirs2[i](1) << " == " << bases2[i](0)
              << ", " << bases2[i](1) << '\n';
  }
  std::cerr << '\n';
  for (const auto& dir : grid_edge_iterator<double, 3>::edge_dirs) {
    std::cerr << dir(0) << ", " << dir(1) << ", " << dir(2) << '\n';
  }
}
//==============================================================================
TEST_CASE("grid_edge_iteration2", "[grid][grid_edge]") {
  size_t counter = 0;
  grid g{linspace{0.0, 2.0, 3}, linspace{0.0, 2.0, 3}};
  for (auto [v0, v1]: g.edges()) {
    std::cerr << v0.indices() << '\n';
    std::cerr << v1.indices() << '\n';
    std::cerr << '\n';
    ++counter;
  }
  REQUIRE(counter == 20);
}
//==============================================================================
TEST_CASE("grid_edge_iteration3", "[grid][grid_edge]") {
  size_t counter = 0;
  grid g{linspace{0.0, 2.0, 3}, linspace{0.0, 2.0, 3}, linspace{0.0, 2.0, 3}};
  std::vector<vec<double,3>> positions;
  std::vector<std::vector<size_t>> lines;
  size_t idx = 0;
  for (auto [v0, v1]: g.edges()) {
    positions.push_back(v0.position());
    positions.push_back(v1.position());
    lines.push_back({idx, idx + 1});
    idx += 2;
    ++counter;
  }
  
  vtk::legacy_file_writer f{"grid_edges_3d.vtk", vtk::POLYDATA};
  f.write_header();
  f.write_points(positions);
  f.write_lines(lines);
  f.close();
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
