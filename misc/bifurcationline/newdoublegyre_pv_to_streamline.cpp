#include "progress.h"
#include "path.h"
#include <tatooine/newdoublegyre.h>
#include <tatooine/line.h>
#include <tatooine/spacetime_field.h>

//==============================================================================
int main() {
  const std::string filepath_acceleration =
      tatooine::misc::bifurcationline::fullpath() +
      "/numerical_spacetime_newdoublegyre_pv_lines_acceleration.vtk";
  const std::string filepath_jerk =
      tatooine::misc::bifurcationline::fullpath() +
      "/numerical_spacetime_newdoublegyre_pv_lines_jerk.vtk";

  tatooine::spacetime_field v{tatooine::numerical::newdoublegyre{}};

  tatooine::misc::bifurcationline::progress(
      v, tatooine::line<double, 3>::read_vtk(filepath_acceleration),
      "newdoublegyre_pv_acc_to_streamline_progression", 20);

  auto jerk_lines = tatooine::line<double, 3>::read_vtk(filepath_jerk);
  auto merged_jerk_lines     = tatooine::merge(jerk_lines, 0.1);
  auto jerk_longest_it = end(merged_jerk_lines);
  auto jerk_longest_length = -std::numeric_limits<double>::max();
  for (auto it = begin(merged_jerk_lines); it!= end(merged_jerk_lines); ++it) {
    if (auto l = it->length(); l > jerk_longest_length) {
      jerk_longest_length = l; 
      jerk_longest_it = it;
    }
  }

  tatooine::misc::bifurcationline::progress(
      v, std::vector{*jerk_longest_it},
      "newdoublegyre_pv_jerk_to_streamline_progression", 20);
}
