#ifndef TATOOINE_DIRECT_VOLUME_RENDERING_H
#define TATOOINE_DIRECT_VOLUME_RENDERING_H
//==============================================================================
#include <tatooine/camera.h>
#include <tatooine/write_png.h>
#include <tatooine/grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number CameraReal, typename Grid, typename Container,
          template <typename> typename Kernel0,
          template <typename> typename Kernel1,
          template <typename> typename Kernel2>
auto direct_volume_rendering(camera<CameraReal> const&                     cam,
                             grid_vertex_property<Grid, Container, Kernel0,
                                                  Kernel1, Kernel2> const& prop,
                             typename Container::value_type const          min,
                             typename Container::value_type const          max,
                             typename Container::value_type const          distance_on_ray) {
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
      auto const r = cam.ray(x, y);
      if (auto const front_intersection = bb.check_intersection(r);
          front_intersection) {
        auto                           cur_t       = front_intersection->t;
        auto                           pos         = r(cur_t);
        typename Container::value_type alpha       = 0;
        typename Container::value_type composition = 0;
        while (bb.is_inside(pos) && alpha < 0.95) {
          pos               = r(cur_t);
          auto const sample = prop.sample(pos);
          auto const color  = (std::max(std::min(sample, max), min) - min) / (max - min);
          std::cerr << color << '\n';
          auto const new_alpha = 0.1;
          composition += color * (1 - alpha);
          alpha += (1 - alpha) * new_alpha;
          cur_t += distance_on_ray;
        }
        rendering.container().at(x, y) = composition * alpha + (1 - alpha) * bg;
      } else {
        //rendering.container().at(x, y) = bg;
      }
    }
  }
  write_png("direct_volume_abc_mag.png", rendering.container().data(),
            cam.plane_width(), cam.plane_height());
  return rendered_image;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
