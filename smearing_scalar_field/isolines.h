#ifndef TATOOINE_SMEARING_ISOLINES_H
#define TATOOINE_SMEARING_ISOLINES_H
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#include <tatooine/line.h>

#include <vector>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
auto create_iso_lines_a(std::vector<uniform_grid_2d<double>> const& grids,
                        double const                                isolevel)
    -> std::vector<std::vector<line<double, 2>>>;
//------------------------------------------------------------------------------
auto create_iso_lines_b(std::vector<uniform_grid_2d<double>> const& grids,
                        double const                                isolevel)
    -> std::vector<std::vector<line<double, 2>>>;
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
#endif
