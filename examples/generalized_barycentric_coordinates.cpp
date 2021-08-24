#include <tatooine/polygon.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/random.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto const x            = std::stod(argv[1]);
  auto const y            = std::stod(argv[2]);
  auto const q            = vec2{x, y};
  auto       polygon_data = std::vector<vec2>{};
  for (int i = 3; i < argc; i += 2) {
    polygon_data.emplace_back(std::stod(argv[i]), std::stod(argv[i + 1]));
  }

  auto       vertex_colors                  = std::vector<real_t>(size(polygon_data));
  auto       rand                           = random::uniform<double>{0,1};
  boost::generate(vertex_colors, [&] {
    return rand();
  });
  for (auto const& c:vertex_colors) {
    std::cerr << c << '\n';
  }
  auto p = polygon2{std::move(polygon_data)};
  for (auto const b : p.barycentric_coordinates(q)) {
    std::cout << b << '\n';
  }
  p.write_vtk("example_polygon.vtk");

  auto sample_grid =
      uniform_rectilinear_grid2{linspace<real_t>{-3, 3, 1000}, linspace<real_t>{-3, 3, 1000}};


  auto& col = sample_grid.scalar_vertex_property("color");
  sample_grid.vertices().iterate_indices([&](auto const... is) {
    auto const x = sample_grid.vertex_at(is...);
    auto       b = p.barycentric_coordinates(x);
    real_t     acc_col = 0;
    for (size_t i = 0; i < b.size(); ++i) {
      acc_col += b[i] * vertex_colors[i];
    }
    col(is...) = acc_col;
  });
  sample_grid.write_vtk("example_polygon_interpolated_color.vtk");
}
