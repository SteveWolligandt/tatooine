#include <tatooine/unstructured_triangular_grid.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  using grid_type     = unstructured_triangular_grid2;
  auto grid           = grid_type{};
  auto const num_vertices_on_inner_circle = 16;
  auto       ts = linspace{0.0, 2 * M_PI, num_vertices_on_inner_circle + 1};
  auto circle_handles = std::vector<grid_type::vertex_handle>{};
  ts.pop_back();
  for (auto const t : ts) {
    circle_handles.push_back(grid.insert_vertex(std::cos(t), std::sin(t)));
  }
  grid.insert_vertex(-2, -2);
  grid.insert_vertex(2, -2);
  grid.insert_vertex(-2, 2);
  grid.insert_vertex(2, 2);
  grid.build_delaunay_mesh();
  grid.write("delaunay_mesh.vtp");
}
