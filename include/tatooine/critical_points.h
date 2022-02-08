#ifndef TATOOINE_CRITICAL_POINTS_H
#define TATOOINE_CRITICAL_POINTS_H
//==============================================================================
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <tatooine/critical_points_bilinear.h>
#include <tatooine/field.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/interpolation.h>
#include <tatooine/sampled_field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename VertexProperty>
auto find_critical_points(sampler<VertexProperty, interpolation::linear,
                                  interpolation::linear> const& s) {
  static_assert(is_vec<typename VertexProperty::value_type>,
                "container of sampler must hold vectors");
  static_assert(VertexProperty::value_type::dimension(0) == 2);
  using namespace boost;
  using namespace adaptors;
  std::vector<typename VertexProperty::grid_t::pos_type> critical_points;
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
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//template <typename Grid, typename Container>
//auto find_critical_points(
//    grid_vertex_property<Grid, Container, interpolation::linear,
//                         interpolation::linear> const& prop) {
//  return find_critical_points(
//      *dynamic_cast<sampler<Grid, Container, interpolation::linear,
//                            interpolation::linear>*>(&prop));
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
