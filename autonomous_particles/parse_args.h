#ifndef TATOOINE_AUTONOMOUS_PARTICLES_PARSE_ARGS_H
#define TATOOINE_AUTONOMOUS_PARTICLES_PARSE_ARGS_H
//==============================================================================
#include <optional>
#include <tatooine/filesystem.h>
//==============================================================================
struct args_t {
  size_t width, height, depth, num_splits, max_num_particles, output_res_x,
      output_res_y, output_res_z;
  double t0, tau, tau_step, min_cond, agranovsky_delta_t;
  bool write_ellipses_to_netcdf;
  bool show_dimensions;
  std::optional<tatooine::filesystem::path> autonomous_particles_file,
      velocity_file;
};
//------------------------------------------------------------------------------
auto parse_args(int argc, char** argv) -> std::optional<args_t>;
//==============================================================================
#endif
