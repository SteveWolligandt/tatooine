#include <tatooine/unstructured_grid.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/random.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto grid = unstructured_grid2{};
  std::vector<unstructured_grid2::vertex_handle> vertex_indices;
  for (int i = 1; i < argc; i += 2) {
    vertex_indices.push_back(
        grid.insert_vertex(std::stod(argv[i]), std::stod(argv[i + 1])));
  }
  grid.insert_cell(vertex_indices);
  grid.insert_cell(0,1,2);

  auto& vertex_colors = grid.scalar_vertex_property("values");
  auto rand          = random::uniform<double>{0, 1};
  boost::generate(vertex_colors, [&] { return rand(); });
  grid.write("example_polygon.vtk");

  auto sample_grid =
      uniform_rectilinear_grid2{linspace<real_type>{-4, 4, 2000}, linspace<real_type>{-4, 4, 2000}};

  auto& col = sample_grid.scalar_vertex_property("color");
  auto sampler = grid.sampler(vertex_colors);
  sample_grid.vertices().iterate_indices([&](auto const... is) {
    auto const x = sample_grid.vertex_at(is...);
    try {
      col(is...) = sampler(x);
    } catch (...) {
      col(is...) = 0.0 / 0.0;
    }
  });
  sample_grid.write_vtk("example_polygon_interpolated_color.vtk");
}
