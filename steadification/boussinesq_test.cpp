#include "datasets.h"
#include "settings.h"
#include <tatooine/streamsurface.h>
#include <tatooine/spacetime_field.h>
//==============================================================================
int main () {
  using namespace tatooine;
  using namespace steadification;
  using seedcurve_t  = line<double, 3>;
  using integrator_t = ode::boost::rungekuttafehlberg78<double, 3>;

  boussinesq v{dataset_dir + "/boussinesq.am"};
  spacetime_field vst{v};
  std::cerr << vst({-0.4, 1, 10}) << '\n';

  seedcurve_t   seedcurve{{{-0.1, 1, 10}, 0}, {{0.1, 1, 10}, 1}};
  integrator_t integrator;
  streamsurface surf{vst, 0, 0, seedcurve, integrator};
  surf.discretize(6, 0.1, -5, 0).write_vtk("boussinesq_test.vtk");
}
