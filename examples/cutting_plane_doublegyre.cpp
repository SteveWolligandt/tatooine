#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/spacetime_vectorfield.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto       v     = analytical::numerical::doublegyre{};
  auto       stv   = spacetime(v);
  auto d = discretize(stv, vec3{0, 0, 0}, vec3{2, 0, 10}, vec3{0, 1, 0}, 1000,
                      200, "dg_cut", 0);
  d.write_vtk("cutting_plane.vtk");
}
