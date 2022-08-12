#include <tatooine/rendering/interactive.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/random.h>
#include <tatooine/streamsurface.h>
#include <tatooine/analytical/abcflow.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  //auto v = analytical::numerical::abcflow{};
  //auto seedcurve = line3{};
  //seedcurve.push_back(-1, 0, 0);
  //seedcurve.push_back(0, 0, 0);
  //seedcurve.push_back(1, 0, 0);
  //seedcurve.compute_parameterization();
  //auto ssf = streamsurface{flowmap(v), 0, seedcurve};
  //auto mesh = unstructured_triangular_grid3{ssf.discretize<naive_discretization>(100, 0.01, -10, 10)};

  auto       mesh = unstructured_triangular_grid2{};
  auto const v0   = mesh.insert_vertex(-4, -4);
  auto const v1   = mesh.insert_vertex(4, -4);
  auto const v2   = mesh.insert_vertex(-4, 4);
  auto const v3   = mesh.insert_vertex(4, 4);
  mesh.insert_triangle(v0, v1, v3);
  mesh.insert_triangle(v0, v3, v2);

  //auto mesh = geometry::discretize(geometry::sphere3{1.0}, 4);

  mesh.write_vtp("ssf.vtp");
  auto & foo = mesh.scalar_vertex_property("foo");
  auto rand = random::uniform{0.0, 1.0};
  for (auto const v : mesh.vertices()) {
    foo[v] = rand();
  }
  rendering::interactive::show(mesh);
}
