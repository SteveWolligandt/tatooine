#include <tatooine/rendering/interactive.h>
#include <tatooine/field_operations.h>
#include <tatooine/streamsurface.h>
#include <tatooine/analytical/doublegyre.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto v2 = analytical::numerical::doublegyre{};
  auto v3 = spacetime(v2);
  auto phi3 = flowmap(v3);
  auto seed = line3{};
  auto v0 = seed.push_back(0.5, 0.1, 0.0);
  auto v1 = seed.push_back(1.5, 0.1, 0.0);
  seed.parameterization()[v0] = 0;
  seed.parameterization()[v1] = 1;
  auto ssf = streamsurface(phi3, seed);
  unstructured_triangular_grid3 mesh = ssf.discretize(10, 0.01, 0, 10);
  rendering::interactive::show(seed, mesh);
}
