#include <tatooine/line.h>
#include <tatooine/newdoublegyre.h>
#include <tatooine/spacetime_field.h>
#include "path.h"
#include "progress.h"

//==============================================================================
int main() {
  const std::string filepath =
      tatooine::misc::bifurcationline::fullpath() + "/straight_line_for_dg.vtk";

  tatooine::spacetime_field v{tatooine::numerical::newdoublegyre{}};

  tatooine::misc::bifurcationline::progress(
      v, tatooine::line<double, 3>::read_vtk(filepath),
      "newdoublegyre_straight_line_to_streamline_progression", 1);
}
