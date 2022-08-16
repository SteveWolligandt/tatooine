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
  auto seedcurve = line3{};
  seedcurve.push_back(-1, 0, 0);
  seedcurve.push_back(1, 0, 0);
  seedcurve.compute_parameterization();
  auto ssf = streamsurface{flowmap(v), 0, seedcurve};
  auto mesh = unstructured_triangular_grid3{ssf.discretize(10, 0.1, 0, 1)};
  mesh.sample_to_vertex_property(v, "velocity");

  rendering::interactive::show(mesh);
}
