#include <tatooine/autonomous_particle.h>
#include <tatooine/analytical/fields/doublegyre.h>

#include <vector>
//==============================================================================
using namespace tatooine;
using namespace detail::autonomous_particle;
//==============================================================================
auto build_example_vertices_of_edgeset() {
  auto edges = edgeset2{};
  //  centers are taken from autonomous_particles
  edges.insert_vertex(0, 0);
  edges.insert_vertex(-1, 0);
  edges.insert_vertex(1, 0);
  edges.insert_vertex(0, 1);
  edges.insert_vertex(0, -1);
  edges.insert_vertex(0, 2);
  edges.insert_vertex(-2, 0);
  edges.insert_vertex(2, 0);
  edges.insert_vertex(3, 0);
  edges.insert_vertex(4, 0);
  edges.insert_vertex(-2, -2);
  edges.insert_vertex(0, -2);
  edges.insert_vertex(3, -2);
  auto map = std::unordered_map<std::size_t, edgeset2::vertex_handle>{
      std::pair{std::size_t(1), edgeset2::vertex_handle{1}},
      std::pair{std::size_t(3), edgeset2::vertex_handle{2}},
      std::pair{std::size_t(4), edgeset2::vertex_handle{4}},
      std::pair{std::size_t(5), edgeset2::vertex_handle{0}},
      std::pair{std::size_t(6), edgeset2::vertex_handle{3}},
      std::pair{std::size_t(12), edgeset2::vertex_handle{5}},
      std::pair{std::size_t(7), edgeset2::vertex_handle{6}},
      std::pair{std::size_t(9), edgeset2::vertex_handle{7}},
      std::pair{std::size_t(10), edgeset2::vertex_handle{8}},
      std::pair{std::size_t(11), edgeset2::vertex_handle{9}},
      std::pair{std::size_t(14), edgeset2::vertex_handle{10}},
      std::pair{std::size_t(15), edgeset2::vertex_handle{11}},
      std::pair{std::size_t(16), edgeset2::vertex_handle{12}}};
  return std::pair{std::move(edges), std::move(map)};
}
//==============================================================================
auto build_example_hierachy_pair_list() {
  auto l = std::vector<hierarchy_pair>{};
  l.emplace_back(0, 18);
  l.emplace_back(1, 0);
  l.emplace_back(2, 0);
  l.emplace_back(3, 0);
  l.emplace_back(4, 2);
  l.emplace_back(5, 2);
  l.emplace_back(6, 2);
  l.emplace_back(7, 18);
  l.emplace_back(8, 18);
  l.emplace_back(9, 8);
  l.emplace_back(10, 8);
  l.emplace_back(11, 8);
  l.emplace_back(12, 17);
  l.emplace_back(13, 17);
  l.emplace_back(14, 13);
  l.emplace_back(15, 13);
  l.emplace_back(16, 13);
  l.emplace_back(17, 17);
  l.emplace_back(18, 17);
  return l;
}
//==============================================================================
auto build_artificial_example() {
  auto const hpl = build_example_hierachy_pair_list();
  auto [es, map]              = build_example_vertices_of_edgeset();
  return std::tuple{std::move(hpl), std::move(es), std::move(map)};
}
//==============================================================================
auto doublegyre_example() {
  auto v              = analytical::fields::numerical::doublegyre{};
  auto uuid_generator = std::atomic_uint64_t{};
  auto const [advected_particles, advected_simple_particles, edges] =
      autonomous_particle2::advect(
          flowmap(v), 0.001, 0, 2,
          rectilinear_grid{linspace{0.8, 1.2, 21},
                           linspace{0.4, 0.6, 11}});

  auto all_advected_discretizations = std::vector<line2>{};
  auto all_initial_discretizations  = std::vector<line2>{};
  for (auto const& p : advected_particles) {
    all_initial_discretizations.push_back(
        discretize(p.initial_ellipse(), 100));
    all_advected_discretizations.push_back(
        discretize(p, 100));
  }
  write(all_initial_discretizations, "post_triangulation_doublegyre_ellipses0.vtk");
  write(all_advected_discretizations,
            "post_triangulation_doublegyre_ellipses1.vtk");
  edges.write("post_triangulation_doublegyre.vtp");
}
//==============================================================================
template <typename Real, std::size_t NumDimensions>
auto print(hierarchy<Real, NumDimensions> const& h, std::string const& tab = "")
    -> void {
  if (h.leafs.empty()) {
    std::cout << tab << h.center << '\n';
  } else {
    std::cout << tab;
    if (h.center != h.center.invalid_idx) {
      std::cout << h.center << ":";
    }
    std::cout << "{\n";
    print(h.leafs.front(), tab + "  ");
    for (std::size_t i = 1; i < size(h.leafs); ++i) {
      print(h.leafs[i], tab + "  ");
    }
    std::cout << tab << "}\n";
  }
}
//==============================================================================
auto main() -> int {
  doublegyre_example();
}