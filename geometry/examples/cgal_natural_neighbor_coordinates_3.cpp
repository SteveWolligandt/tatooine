#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <tatooine/cgal/delaunay_triangulation.h>
#include <tatooine/cgal/natural_neighbor_coordinates.h>
#include <tatooine/demangling.h>

using namespace tatooine;
auto main() -> int {
  using kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using triangulation_type = cgal::delaunay_triangulation_with_info<
      3, int, kernel,
      cgal::delaunay_triangulation_simplex_base_with_circumcenter<3, kernel>>;
  std::cout << type_name<triangulation_type>() << '\n';
  using point_type = triangulation_type::Point;
  auto points      = std::vector<std::pair<point_type, int>>{};
  points.emplace_back(point_type(0, 0, 0), 1);
  points.emplace_back(point_type(1, 0, 0), 2);
  points.emplace_back(point_type(0, 1, 0), 3);
  points.emplace_back(point_type(1, 1, 0), 4);
  points.emplace_back(point_type(0, 0, 1), 5);
  points.emplace_back(point_type(1, 0, 1), 6);
  points.emplace_back(point_type(0, 1, 1), 7);
  points.emplace_back(point_type(1, 1, 1), 8);

  auto triangulation = triangulation_type{begin(points), end(points)};
  auto num_faces     = std::size_t{};
  for (auto it = triangulation.finite_cells_begin();
       it != triangulation.finite_cells_end(); ++it) {
    ++num_faces;
    std::cout << it->vertex(0)->info() << ", " << it->vertex(1)->info() << ", "
              << it->vertex(2)->info() << ", " << it->vertex(3)->info() << '\n';
  }
  std::cout << num_faces << '\n';

  auto [x, nncs] = cgal::natural_neighbor_coordinates<
      3, triangulation_type::Geom_traits,
      triangulation_type::Triangulation_data_structure>(
      triangulation, triangulation_type::Point{0.5, 0.5, 0.5});
  for (auto const& [v, w] : nncs) {
    std::cout << w << ", " << v->info() << '\n';
  }
}
