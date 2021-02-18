#include <tatooine/counterexample_sadlo.h>
#include <tatooine/line.h>
#include <tatooine/spacetime_field.h>
#include <array>
#include <tatooine/filesystem.h>
#include <list>
#include "progress.h"

using namespace tatooine;

const std::string fullpath = FULLPATH;
const std::string filepath_acceleration =
    fullpath +
    "/numerical_spacetime_counterexample_sadlo_pv_lines_acceleration.vtk";
const std::string filepath_jerk =
    fullpath + "/numerical_spacetime_counterexample_sadlo_pv_lines_jerk.vtk";


//==============================================================================
int main() {
  spacetime_field v{numerical::counterexample_sadlo{}};
  tatooine::misc::bifurcationline::progress(
      v,
      std::vector{
          merge(line<double, 3>::read_vtk(filepath_acceleration), 1).front()},
      "counterexample_pv_acc", 30);

  auto jerk_lines    = merge(filter_length(line<double, 3>::read_vtk(filepath_jerk), 1), 0.5);
  auto max_length    = -std::numeric_limits<double>::max();
  auto max_length_it = end(jerk_lines);
  for (auto it = begin(jerk_lines); it != end(jerk_lines); ++it) {
    if (auto l = it->length(); max_length < l) {
      max_length = l;
      max_length_it = it;
    }
  }
  tatooine::misc::bifurcationline::progress(v, std::vector{*max_length_it}, "counterexample_pv_jerk", 30);
}
