#include "progress.h"
#include "path.h"
#include <tatooine/doublegyre.h>
#include <tatooine/line.h>
#include <tatooine/spacetime_field.h>

//==============================================================================
int main() {
  const std::string filepath =
      tatooine::misc::bifurcationline::fullpath() + "/straight_line_for_dg.vtk";

  tatooine::spacetime_field v{tatooine::numerical::doublegyre{}};

  tatooine::misc::bifurcationline::progress(
      v, tatooine::line<double, 3>::read_vtk(filepath),
      "doublegyre_straight_line_to_streamline_progression", 1);
}
