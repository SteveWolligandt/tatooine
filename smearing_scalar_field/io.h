#ifndef TATOOINE_SMEARING_IO_H
#define TATOOINE_SMEARING_IO_H
//==============================================================================
#include <tatooine/grid.h>

#include <filesystem>
#include <vector>
//==============================================================================
namespace tatooine::smearing {
//==============================================================================
auto read_ascii(std::filesystem::path const& file_path, int& res_x, int& res_y,
                int& res_t, double& min_x, double& min_y, double& min_t,
                double& extent_x, double& extent_y, double& extent_t)
    -> std::vector<uniform_grid_2d<double>>;
//------------------------------------------------------------------------------
auto read_binary(std::filesystem::path const& file_path, int& res_x, int& res_y,
                 int& res_t, double& min_x, double& min_y, double& min_t,
                 double& extent_x, double& extent_y, double& extent_t)
    -> std::vector<uniform_grid_2d<double>>;
//------------------------------------------------------------------------------
auto write_ascii(std::filesystem::path const&                file_path,
                 std::vector<uniform_grid_2d<double>> const& grids, int res_x,
                 int res_y, int res_t, double min_x, double min_y, double min_t,
                 double extent_x, double extent_y, double extent_t) -> void;
//------------------------------------------------------------------------------
auto write_binary(std::filesystem::path const&                file_path,
                  std::vector<uniform_grid_2d<double>> const& grids, int res_x,
                  int res_y, int res_t, double min_x, double min_y,
                  double min_t, double extent_x, double extent_y,
                  double extent_t) -> void;
//==============================================================================
}  // namespace tatooine::smearing
//==============================================================================
#endif
