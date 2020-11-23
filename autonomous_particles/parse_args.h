#ifndef TATOOINE_AUTONOMOUS_PARTICLES_PARSE_ARGS_H
#define TATOOINE_AUTONOMOUS_PARTICLES_PARSE_ARGS_H
//==============================================================================
#include <optional>
//==============================================================================
struct args_t {
  size_t width, height, depth, num_splits, max_num_particles;
  double t0, tau, tau_step, min_cond;
};
//------------------------------------------------------------------------------
auto parse_args(int argc, char** argv) -> std::optional<args_t>;
//==============================================================================
#endif
