#ifndef TATOOINE_DIRECT_VOLUME_RENDERING_H
#define TATOOINE_DIRECT_VOLUME_RENDERING_H
//==============================================================================
#include <tatooine/camera.h>
#include <tatooine/demangling.h>
#include <tatooine/grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number CameraReal, real_number BBReal,
          regular_invocable<vec<BBReal, 3>> DataEvaluator,
          regular_invocable<vec<BBReal, 3>> DomainCheck>
auto direct_volume_rendering(camera<CameraReal> const&     cam,
                             boundingbox<BBReal, 3> const& bb,
                             DataEvaluator&&               data_evaluator,
                             DomainCheck&&                 domain_check,
                             real_number auto const        min,
                             real_number auto const        max,
                             real_number auto const        distance_on_ray) {
  using pos_t      = vec<BBReal, 3>;
  using value_type = std::invoke_result_t<DataEvaluator, pos_t>;
  static_assert(std::is_arithmetic_v<value_type>,
                "DataEvaluator must return scalar type.");
  grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, 1.0, cam.plane_width()},
      linspace<CameraReal>{0.0, 1.0, cam.plane_height()}};
  auto& rendering =
      rendered_image.template add_contiguous_vertex_property<value_type>(
          "rendering");

  value_type const bg = 1;
#pragma omp parallel for
  for (size_t y = 0; y < cam.plane_height(); ++y) {
    for (size_t x = 0; x < cam.plane_width(); ++x) {
      auto r = cam.ray(x, y);
      r.normalize();
      if (auto const front_intersection = bb.check_intersection(r);
          front_intersection) {
        value_type                  accumuluated_alpha = 1;
        value_type                  accumulated_color  = 0;
        auto                        cur_t              = front_intersection->t;
        pos_t cur_pos            = r(cur_t);
        if (!bb.is_inside(cur_pos)) {
          cur_t += 1e-6;
          cur_pos = r(cur_t);
        }

        while (bb.is_inside(cur_pos) && 1 - accumuluated_alpha < 0.95) {
          if (domain_check(cur_pos)) {
            auto const sample = data_evaluator(cur_pos);
            auto       sample_color =
                (std::max<value_type>(std::min<value_type>(sample, max), min) -
                 min) /
                (max - min);
            auto sample_alpha = std::max<value_type>(0, sample_color - 0.2);

            sample_alpha = 1.0 - std::exp(-0.5 * sample_alpha);
            sample_color *= sample_alpha;

            accumulated_color += sample_color * accumuluated_alpha;
            accumuluated_alpha *= 1 - sample_alpha;
            cur_t += distance_on_ray;
            cur_pos = r(cur_t);
          }
        }
        rendering.container().at(x, y) =
            accumulated_color * (1 - accumuluated_alpha) +
            accumuluated_alpha * bg;
      } else {
         rendering.container().at(x, y) = bg;
      }
    }
  }
  return rendered_image;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number CameraReal, real_number BBReal, typename S,
          typename SReal>
auto direct_volume_rendering(camera<CameraReal> const&       cam,
                             boundingbox<BBReal, 3> const&   bb,
                             scalarfield<S, SReal, 3> const& s,
                             real_number auto const          t,
                             real_number auto const          min,
                             real_number auto const          max,
                             real_number auto const          distance_on_ray) {
  return direct_volume_rendering(
      cam, bb,
      [&](auto const& x) {
        return s(x, t);
      },
      [&](auto const& x) {
        return s.in_domain(x, t);
      },
      min, max, distance_on_ray);
}  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   // - -
template <real_number CameraReal, typename Grid, typename Container,
          template <typename> typename Kernel0,
          template <typename> typename Kernel1,
          template <typename> typename Kernel2>
auto direct_volume_rendering(
    camera<CameraReal> const& cam,
    grid_vertex_property<Grid, Container, Kernel0, Kernel1, Kernel2> const&
                                         prop,
    typename Container::value_type const min,
    typename Container::value_type const max,
    typename Container::value_type const distance_on_ray) {
  return direct_volume_rendering(
      cam, prop.grid().boundingbox(),
      [&](auto const& x) {
        return prop.sample(x);
      },
      [](auto const&) {
        return true;
      },
      min, max, distance_on_ray);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
