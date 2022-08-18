#ifndef TATOOINE_CRITICAL_POINTS_H
#define TATOOINE_CRITICAL_POINTS_H
//==============================================================================
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <tatooine/critical_points_bilinear.h>
#include <tatooine/field.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/interpolation.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Grid, typename Real, bool HasNonConstReference>
auto find_critical_points(
    detail::rectilinear_grid::vertex_property_sampler<
        detail::rectilinear_grid::typed_vertex_property_interface<Grid, vec<Real, 2>,
                                                        HasNonConstReference>,
        interpolation::linear, interpolation::linear> const& s) {
  using namespace boost;
  using namespace adaptors;
  auto critical_points = std::vector<typename Grid::pos_type>{};
  // iterate over each grid cell
  for (size_t y = 0; y < s.grid().template size<1>() - 1; ++y) {
    for (size_t x = 0; x < s.grid().template size<0>() - 1; ++x) {
      copy(
          detail::solve_bilinear(s.data_at(x, y), s.data_at(x + 1, y),
                                 s.data_at(x, y + 1), s.data_at(x + 1, y + 1)) |
              transformed([x, y, &s](const auto& st) {
                return vec{(1 - st(0)) * s.grid().template dimension<0>()[x] +
                               (st(0)) * s.grid().template dimension<0>()[x + 1],
                           (1 - st(1)) * s.grid().template dimension<1>()[y] +
                               (st(1)) * s.grid().template dimension<1>()[y + 1]};
              }),
          std::back_inserter(critical_points));
    }
  }
  return critical_points;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
