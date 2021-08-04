#ifndef TATOOINE_SMEARING_IO_H
#define TATOOINE_SMEARING_IO_H
//==============================================================================
#include <tatooine/rectilinear_grid.h>

#include <tatooine/filesystem.h>
#include <vector>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
auto read_ascii(filesystem::path const& file_path, int& res_x, int& res_y,
                int& res_t, double& min_x, double& min_y, double& min_t,
                double& extent_x, double& extent_y, double& extent_t)
    -> std::vector<uniform_rectilinear_grid_2d<double>>;
//------------------------------------------------------------------------------
auto read_binary(filesystem::path const& file_path, int& res_x, int& res_y,
                 int& res_t, double& min_x, double& min_y, double& min_t,
                 double& extent_x, double& extent_y, double& extent_t)
    -> std::vector<uniform_rectilinear_grid_2d<double>>;
//------------------------------------------------------------------------------
auto write_ascii(filesystem::path const&                file_path,
                 std::vector<uniform_rectilinear_grid_2d<double>> const& grids, int res_x,
                 int res_y, int res_t, double min_x, double min_y, double min_t,
                 double extent_x, double extent_y, double extent_t) -> void;
//------------------------------------------------------------------------------
auto write_binary(filesystem::path const&                file_path,
                  std::vector<uniform_rectilinear_grid_2d<double>> const& grids, int res_x,
                  int res_y, int res_t, double min_x, double min_y,
                  double min_t, double extent_x, double extent_y,
                  double extent_t) -> void;
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
#endif
