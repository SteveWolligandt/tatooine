#ifndef TATOOINE_BOUNDARY_SWITCH_H
#define TATOOINE_BOUNDARY_SWITCH_H
//==============================================================================
#include "field.h"
#include "grid_sampler.h"
#include "interpolation.h"
#include "sampled_field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
auto find_boundary_switch_points(
    const grid_sampler<Real, 2, vec<Real, 2>, interpolation::linear,
                       interpolation::linear>& sampler) {
  const size_t              left   = 0;
  const size_t              right  = sampler.size(0) - 1;
  const size_t              bottom = 0;
  const size_t              top    = sampler.size(1) - 1;
  std::vector<vec<Real, 2>> boundary_switch_points;

  // iterate over each boundary grid cell in x-direction
  for (size_t i = 0; i < sampler.size(0) - 1; ++i) {
    if ((sampler[i][bottom](1) <= 0 && sampler[i + 1][bottom](1) > 0) ||
        (sampler[i][bottom](1) >= 0 && sampler[i + 1][bottom](1) < 0)) {
      const Real t =
         - sampler[i][bottom](1) / (sampler[i + 1][bottom](1) - sampler[i][bottom](1));
      boundary_switch_points.emplace_back(
          sampler.dimension(0)[i] * (1 - t) + sampler.dimension(0)[i + 1] * t,
          sampler.dimension(1)[bottom]);
    }

    if ((sampler[i][top](1) <= 0 && sampler[i + 1][top](1) > 0) ||
        (sampler[i][top](1) >= 0 && sampler[i + 1][top](1) < 0)) {
      const Real t =
         - sampler[i][top](1) / (sampler[i + 1][top](1) - sampler[i][top](1));
      boundary_switch_points.emplace_back(
          sampler.dimension(0)[i] * (1 - t) + sampler.dimension(0)[i + 1] * t,
          sampler.dimension(1)[top]);
    }
  }
  // iterate over each boundary grid cell in y-direction
  for (size_t i = 0; i < sampler.size(1) - 1; ++i) {
    if ((sampler[left][i](0) <= 0 && sampler[left][i + 1](0) > 0) ||
        (sampler[left][i](0) >= 0 && sampler[left][i + 1](0) < 0)) {
      const Real t =
        -  sampler[left][i](0) / (sampler[left][i + 1](0) - sampler[left][i](0));
      boundary_switch_points.emplace_back(
          sampler.dimension(0)[left],
          sampler.dimension(1)[i] * (1 - t) + sampler.dimension(1)[i + 1] * t);
    }
    if ((sampler[right][i](0) <= 0 && sampler[right][i + 1](0) > 0) ||
        (sampler[right][i](0) >= 0 && sampler[right][i + 1](0) < 0)) {
      const Real t =
        -  sampler[right][i](0) / (sampler[right][i + 1](0) - sampler[right][i](0));
      boundary_switch_points.emplace_back(
          sampler.dimension(0)[right],
          sampler.dimension(1)[i] * (1 - t) + sampler.dimension(1)[i + 1] * t);
    }
  }
  return boundary_switch_points;
}
//------------------------------------------------------------------------------
template <typename Real>
auto find_boundary_switch_points(
    const sampled_field<
        grid_sampler<Real, 2, vec<Real, 2>, interpolation::linear,
                     interpolation::linear>,
        Real, 2, 2>& v) {
  return find_boundary_switch_points(v.sampler());
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
