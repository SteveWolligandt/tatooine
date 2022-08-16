#include <tatooine/rendering/interactive.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/field_operations.h>
#include <tatooine/random.h>
#include <tatooine/streamsurface.h>
#include <tatooine/analytical/abcflow.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v = analytical::numerical::abcflow{};
  //auto seedcurve = line3{};
  //seedcurve.push_back(-1, 0, 0);
  //seedcurve.push_back(0, 0, 0);
  //seedcurve.push_back(1, 0, 0);
  //seedcurve.compute_parameterization();
  //auto ssf = streamsurface{flowmap(v), 0, seedcurve};
  //auto mesh = unstructured_triangular_grid3{ssf.discretize<naive_discretization>(100, 0.01, -10, 10)};

  auto       mesh = unstructured_triangular_grid2{};
  auto rand = random::uniform{0.0, 1.0};
  for (std::size_t i = 0 ; i < 100; ++i) {
    mesh.insert_vertex(rand(), rand());
  }
  mesh.build_delaunay_mesh();

  auto mesh2 = geometry::discretize(geometry::sphere3{.5, vec3{1,0,0}}, 4);
  mesh2.sample_to_vertex_property(euclidean_length(v), "mag");

  auto & foo = mesh.scalar_vertex_property("foo");
  auto rand2 = random::uniform{0.0, 1.0};
  mesh.write_vtp("ssf.vtp");
  for (auto const v : mesh.vertices()) {
    foo[v] = rand2();
  }
  rendering::interactive::show(mesh, mesh2);
}
