#ifndef TATOOINE_AUTONOMOUS_PARTICLES_PARSE_ARGS_H
#define TATOOINE_AUTONOMOUS_PARTICLES_PARSE_ARGS_H
//==============================================================================
#include <optional>
//==============================================================================
struct args_t {
  size_t width, height, depth, num_splits, max_num_particles, output_res_x,
      output_res_y;
  double t0, tau, tau_step, min_cond;
  bool write_ellipses_to_netcdf;
};
//------------------------------------------------------------------------------
auto parse_args(int argc, char** argv) -> std::optional<args_t>;
//==============================================================================
#endif
