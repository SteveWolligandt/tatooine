#ifndef TATOOINE_AUTONOMOUS_PARTICLES_ADVECT_H
#define TATOOINE_AUTONOMOUS_PARTICLES_ADVECT_H
//==============================================================================
#include <tatooine/autonomous_particle.h>

#include "split_behavior.h"
//==============================================================================
namespace tatooine::autonomous_particles {
//==============================================================================
template <typename Real>
auto advect(autonomous_particle<Real, 2> const& particle,
            split_behavior_t const                           s) {
  using particle_type = autonomous_particle<Real, 2>;
  switch (s) {
    case split_behavior_t::two_splits:
      return particle.advect<particle_type::split_behaviors::two_splits>(
          phi, args.step_width, t + stepwidth);
    default:
    case split_behavior_t::three_splits:
      return particle.advect<
          particle_type::split_behaviors::three_splits>(
          phi, args.step_width, t + stepwidth);
    case split_behavior_t::three_in_square_splits:
      return particle_type::advect<
          particle.split_behaviors::three_in_square_splits>(
          phi, args.step_width, t + stepwidth);
    case split_behavior_t::centered_four:
      return particle.advect<
          particle_type::split_behaviors::centered_four>(
          phi, args.step_width, t + stepwidth);
  }
}
//==============================================================================
template <typename Real>
auto advect(std::vector<autonomous_particle<Real, 2>> const& particles,
            split_behavior_t const                           s) {
  using particle_type = autonomous_particle<Real, 2>;
  switch (s) {
    case split_behavior_t::two_splits:
      return particle_type::advect<particle_type::split_behaviors::two_splits>(
          phi, args.step_width, t + stepwidth, particles);
    default:
    case split_behavior_t::three_splits:
      return particle_type::advect<
          particle_type::split_behaviors::three_splits>(
          phi, args.step_width, t + stepwidth, particles);
    case split_behavior_t::three_in_square_splits:
      return particle_type::advect<
          particle_type::split_behaviors::three_in_square_splits>(
          phi, args.step_width, t + stepwidth, particles);
    case split_behavior_t::centered_four:
      return particle_type::advect<
          particle_type::split_behaviors::centered_four>(
          phi, args.step_width, t + stepwidth, particles);
  }
}
//==============================================================================
template <typename Real>
auto advect(autonomous_particle<Real, 3> const& particle,
            split_behavior_t const                           s) {
  using particle_type = autonomous_particle<Real, 3>;
  switch (s) {
    default:
    case split_behavior_t::three_splits:
      return particle.advect<particle_type::split_behaviors::three_splits>(
          phi, args.step_width, args.tau);
  }
}
//------------------------------------------------------------------------------
template <typename Real>
auto advect(std::vector<autonomous_particle<Real, 3>> const& particles,
            split_behavior_t const                           s) {
  using particle_type = autonomous_particle<Real, 3>;
  switch (s) {
    default:
    case split_behavior_t::three_splits:
      return particle_type::advect<particle_type::split_behaviors::three_splits>(
          phi, args.step_width, args.tau, particles);
  }
}
//==============================================================================
}  // namespace tatooine::autonomous_particles
//==============================================================================
