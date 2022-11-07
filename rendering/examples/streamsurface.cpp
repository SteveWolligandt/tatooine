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
  auto ssf = streamsurface(v3, line3{vec3{0.9, 0.5, 0.0}, vec3{1.1, 0.5, 0.0}});
  auto mesh = ssf.discretize(3, 0.1, 0, 10);
  rendering::interactive::show(mesh);
}
