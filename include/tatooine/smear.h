#ifndef TATOOINE_SMEAR_H
#define TATOOINE_SMEAR_H
//==============================================================================
#include <tatooine/geometry/sphere.h>
#include <tatooine/grid.h>
#include <tatooine/interpolation.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// Stores a smeared version of ping_field into pong_field.
/// For each point of a grid go in backward directions and sample field there.
/// Afterwards create the interpolation factor depending of position and time.
template <template <typename>
          typename InterpolationKernel = interpolation::cubic,
          typename PingGrid, real_number PingReal, typename PongGrid,
          real_number PongReal, real_number SphereReal, real_number DirReal>
auto smear(typed_multidim_property<PingGrid, PingReal> const& ping_field,
           typed_multidim_property<PongGrid, PongReal>&       pong_field,
           geometry::sphere<SphereReal, 2> const&             s,
           real_number auto const                             inner_radius,
           real_number auto const                             temporal_range,
           real_number auto const current_time, real_number auto const t0,
           vec<DirReal, 2> const& dir) {
  // create a sampler of the ping_field
  auto sampler = ping_field.template sampler<InterpolationKernel>();

  ping_field.grid().parallel_loop_over_vertex_indices([&](auto const... is) {
    if (std::abs(t0 - current_time) > temporal_range) {
      auto const sampled_current = ping_field(is...);
      pong_field(is...)          = sampled_current;
      return;
    }
    auto const current_pos               = ping_field.grid()(is...);
    auto const offset_pos                = current_pos - dir;
    auto const distance_to_sphere_origin = distance(current_pos, s.center());
    if (distance_to_sphere_origin < s.radius()) {
      auto const s_x = [&]() -> double {
        if (distance_to_sphere_origin <= inner_radius) {
          return 1;
        }
        if (distance_to_sphere_origin > s.radius()) {
          return 0;
        }
        return (distance_to_sphere_origin - s.radius()) /
               (inner_radius - s.radius());
      }();
      auto const lambda_s = s_x * s_x * s_x + 3 * s_x * s_x * (1 - s_x);
      auto const s_t      = [&]() -> double {
        if (auto const t_diff = std::abs(current_time - t0);
            t_diff < temporal_range) {
          return 1 - t_diff / temporal_range;
        }
        return 0;
      }();
      auto const lambda_t = s_t * s_t * s_t + 3 * s_t * s_t * (1 - s_t);
      assert(lambda_s >= 0 && lambda_s <= 1);
      auto const lambda = lambda_s * lambda_t;
      if (!ping_field.grid().bounding_box().is_inside(current_pos) ||
          !ping_field.grid().bounding_box().is_inside(offset_pos)) {
        pong_field(is...) = 0.0 / 0.0;
      } else {
        auto const sampled_current = sampler(current_pos);
        auto const sampled_smeared = sampler(offset_pos);
        pong_field(is...) =
            sampled_current * (1 - lambda) + sampled_smeared * lambda;
      }
    }
  });
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
