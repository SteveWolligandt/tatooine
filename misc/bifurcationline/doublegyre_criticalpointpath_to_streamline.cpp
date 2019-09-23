#include <tatooine/doublegyre.h>
#include <tatooine/line.h>
#include <tatooine/spacetime_field.h>
#include "path.h"
#include "progress.h"

//==============================================================================
using namespace tatooine;
using namespace tatooine::numerical;

//==============================================================================
int main() {
  const std::string filepath = tatooine::misc::bifurcationline::fullpath() +
                               "/doublegyre_criticalpointpath.vtk";
  spacetime_field v{doublegyre{}};

  tatooine::misc::bifurcationline::progress(
      v, tatooine::line<double, 3>::read_vtk(filepath),
      "doublegyre_criticalpointpath_to_streamline_progression", 1);
}
