#include "isolines.h"

#include <tatooine/isolines.h>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
auto create_iso_lines(std::vector<uniform_rectilinear_grid_2d<double>> const& grids,
                      double const isolevel, std::string const& name)
    -> std::vector<std::vector<line<double, 2>>> {
  std::vector<std::vector<line<double, 2>>> ls;
  for (size_t i = 0; i < size(grids); ++i) {
    auto const& prop = grids[i].vertex_property<double>(name);
    ls.push_back(isolines(prop, isolevel));
  }
  return ls;
}
//------------------------------------------------------------------------------
auto create_iso_lines_a(std::vector<uniform_rectilinear_grid_2d<double>> const& grids,
                        double const                                isolevel)
    -> std::vector<std::vector<line<double, 2>>> {
  return create_iso_lines(grids, isolevel, "a");
}
//------------------------------------------------------------------------------
auto create_iso_lines_b(std::vector<uniform_rectilinear_grid_2d<double>> const& grids,
                        double const                                isolevel)
    -> std::vector<std::vector<line<double, 2>>> {
  return create_iso_lines(grids, isolevel, "b");
}
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
