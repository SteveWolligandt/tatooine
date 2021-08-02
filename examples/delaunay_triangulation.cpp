#include <tatooine/unstructured_triangular_grid.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto mesh           = unstructured_triangular_grid_2{};
  auto ts             = linspace{0.0, 2 * M_PI, 9};
  auto constraints    = std::vector<std::pair<unstructured_triangular_grid_2::vertex_handle,
                                           unstructured_triangular_grid_2::vertex_handle>>{};
  auto circle_handles = std::vector<unstructured_triangular_grid_2::vertex_handle>{};
  for (auto t_it = begin(ts); t_it != prev(end(ts)); ++t_it) {
    auto const t = *t_it;
    circle_handles.push_back(mesh.insert_vertex(std::cos(t), std::sin(t)));
  }
  for (size_t i = 0; i < size(circle_handles) - 1; ++i) {
    constraints.emplace_back(circle_handles[i], circle_handles[i + 1]);
  }
  mesh.insert(-2, -2);
  mesh.insert(2, -2);
  mesh.insert(-2, 2);
  mesh.insert(2, 2);
  mesh.build_constrained_delaunay_mesh(constraints);
  mesh.write_vtk("constrained_delaunay.vtk");
}
