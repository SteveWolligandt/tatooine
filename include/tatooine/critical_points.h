#ifndef TATOOINE_CRITICAL_POINTS_H
#define TATOOINE_CRITICAL_POINTS_H

#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include "critical_points_bilinear.h"
#include "field.h"
#include "grid_sampler.h"
#include "interpolation.h"
#include "sampled_field.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real>
auto find_critical_points(
    const grid_sampler<Real, 2, vec<Real, 2>, interpolation::linear,
                       interpolation::linear>& sampler) {
  using namespace boost;
  using namespace adaptors;
  std::vector<vec<Real, 2>> critical_points;
  // iterate over each grid cell
  for (size_t y = 0; y < sampler.size(1) - 1; ++y) {
    for (size_t x = 0; x < sampler.size(0) - 1; ++x) {
      copy(detail::solve_bilinear(sampler[x][y], sampler[x + 1][y],
                                  sampler[x][y + 1], sampler[x + 1][y + 1]) |
               transformed([x, y, &sampler](const auto& st) {
                 return vec{(1 - st(0)) * sampler.dimension(0)[x] +
                                (st(0)) * sampler.dimension(0)[x+1],
                            (1 - st(1)) * sampler.dimension(1)[y] +
                                (st(1)) * sampler.dimension(1)[y+1]};
               }),
           std::back_inserter(critical_points));
    }
  }
  return critical_points;
}
//------------------------------------------------------------------------------
template <typename Real>
auto find_critical_points(
    const sampled_field<
        grid_sampler<Real, 2, vec<Real, 2>, interpolation::linear,
                     interpolation::linear>,
        Real, 2, 2>& v) {
  return find_critical_points(v.sampler());
}
//==============================================================================
}
//==============================================================================

#endif
