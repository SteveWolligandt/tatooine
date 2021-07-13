#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/spacetime_vectorfield.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto       v     = analytical::fields::numerical::doublegyre{};
  auto       stv   = spacetime(v);
  auto const basis = mat<real_t, 2, 3>{};
  basis.col(0)     = normalize(vec{2, 0, 10});
  basis.col(1)     = normalize(vec{0, 1, 0});
  auto d = discretize(stv, basis, vec3::zeros(), vec2{length(vec2{2, 10}), 1},
                      1000, 200, "dg_cut", 0);
  d.write_vtk("cutting_plane.vtk", basis);
}
