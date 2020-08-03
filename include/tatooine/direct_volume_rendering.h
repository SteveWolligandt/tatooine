#ifndef TATOOINE_DIRECT_VOLUME_RENDERING_H
#define TATOOINE_DIRECT_VOLUME_RENDERING_H
//==============================================================================
#include <tatooine/camera.h>
#include <tatooine/grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
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
  static_assert(std::is_arithmetic_v<typename Container::value_type>,
                "Container must hold scalar type");
  grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, 1.0, cam.plane_width()},
      linspace<CameraReal>{0.0, 1.0, cam.plane_height()}};
  auto& rendering = rendered_image.template add_contiguous_vertex_property<
      typename Container::value_type>("rendering");
  auto const bb = prop.grid().boundingbox();

  typename Container::value_type bg = 1;
#pragma omp parallel for
  for (size_t y = 0; y < cam.plane_height(); ++y) {
    for (size_t x = 0; x < cam.plane_width(); ++x) {
      auto r = cam.ray(x, y);
      r.normalize();
      if (auto const front_intersection = bb.check_intersection(r);
          front_intersection) {
        typename Container::value_type accumuluated_alpha = 1;
        typename Container::value_type accumulated_color  = 0;
        auto                           cur_t   = front_intersection->t;
        typename decltype(r)::pos_t    cur_pos = r(cur_t);
        if (!bb.is_inside(cur_pos)) {
          cur_t += 1e-6;
          cur_pos = r(cur_t);
        }

        while (bb.is_inside(cur_pos) && 1 - accumuluated_alpha < 0.95) {
          auto const sample = prop.sample(cur_pos);
          auto       sample_color =
              (std::max(std::min(sample, max), min) - min) / (max - min);
          auto sample_alpha = sample_color;

          sample_alpha = 1.0 - std::exp(-0.5 * sample_alpha);
          sample_color *= sample_alpha;

          accumulated_color += sample_color * accumuluated_alpha;
          accumuluated_alpha *= 1 - sample_alpha;
          cur_t += distance_on_ray;
          cur_pos = r(cur_t);
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
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
