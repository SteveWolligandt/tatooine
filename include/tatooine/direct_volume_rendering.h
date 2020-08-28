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
          regular_invocable<vec<BBReal, 3>> DomainCheck,
          regular_invocable<std::invoke_result_t<DataEvaluator, vec<BBReal, 3>>>
              ColorScale,
          regular_invocable<std::invoke_result_t<DataEvaluator, vec<BBReal, 3>>>
              AlphaScale>
auto direct_volume_rendering(
    camera<CameraReal> const& cam, boundingbox<BBReal, 3> const& bb,
    DataEvaluator&& data_evaluator, DomainCheck&& domain_check,
    real_number auto const min, real_number auto const max,
    real_number auto const distance_on_ray, ColorScale&& color_scale,
    AlphaScale&& alpha_scale,
    std::invoke_result_t<
        ColorScale, std::invoke_result_t<DataEvaluator, vec<BBReal, 3>>> const&
        bg_color = {}) {
  using pos_t      = vec<BBReal, 3>;
  using value_type = std::invoke_result_t<DataEvaluator, pos_t>;
  using color_type = std::invoke_result_t<ColorScale, value_type>;
  using alpha_type = std::invoke_result_t<AlphaScale, value_type>;
  static_assert(std::is_arithmetic_v<value_type>,
                "DataEvaluator must return scalar type.");
  static_assert(std::is_arithmetic_v<color_type> || is_vector_v<color_type>,
                "ColorScale must return scalar type or tatooine::vec.");
  static_assert(std::is_floating_point_v<alpha_type>,
                "AlphaScale must return floating point number.");
  grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, 1.0, cam.plane_width()},
      linspace<CameraReal>{0.0, 1.0, cam.plane_height()}};
  auto& rendering =
      rendered_image.template add_contiguous_vertex_property<color_type>(
          "rendering");

  std::vector<std::tuple<ray<CameraReal, 3>, BBReal, size_t, size_t>> rays;
  for (size_t y = 0; y < cam.plane_height(); ++y) {
    for (size_t x = 0; x < cam.plane_width(); ++x) {
      rendering.container().at(x, y) = bg_color;
      auto r                         = cam.ray(x, y);
      r.normalize();
      if (auto const i = bb.check_intersection(r); i) {
        rays.push_back(std::tuple{r, i->t, x, y});
      }
    }
  }
#pragma omp parallel for
  for (size_t i = 0; i < rays.size(); ++i) {
    auto const [r, t, x, y]      = rays[i];
    alpha_type accumulated_alpha = 0;
    color_type accumulated_color{};
    auto       cur_t   = t;
    pos_t      cur_pos = r(cur_t);
    if (!bb.is_inside(cur_pos)) {
      cur_t += 1e-6;
      cur_pos = r(cur_t);
    }

    while (bb.is_inside(cur_pos) && accumulated_alpha < 0.95) {
      if (domain_check(cur_pos)) {
        auto const sample =
            data_evaluator(cur_pos);
        auto const normalized_sample =
            (std::clamp<value_type>(sample, min, max) - min) / (max - min);
        auto const sample_color =
            color_scale(normalized_sample);
        auto const sample_alpha =
            std::clamp<alpha_type>(alpha_scale(normalized_sample), 0, 1);

        accumulated_color +=
            (1 - accumulated_alpha) * sample_alpha * sample_color;
        accumulated_alpha += (1 - accumulated_alpha) * sample_alpha;
      }
      cur_t += distance_on_ray;
      cur_pos = r(cur_t);
    }
    rendering.container().at(x, y) = accumulated_color * accumulated_alpha +
                                     bg_color * (1 - accumulated_alpha);
  }
  return rendered_image;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number CameraReal, real_number      BBReal, typename S,
          typename SReal, regular_invocable<SReal> ColorScale,
          regular_invocable<SReal> AlphaScale>
auto direct_volume_rendering(
    camera<CameraReal> const& cam, boundingbox<BBReal, 3> const& bb,
    scalarfield<S, SReal, 3> const& s, real_number auto const t,
    real_number auto const min, real_number auto const max,
    real_number auto const distance_on_ray, ColorScale&& color_scale,
    AlphaScale&&                                   alpha_scale,
    std::invoke_result_t<ColorScale, SReal> const& bg_color = {}) {
  return direct_volume_rendering(
      cam, bb,
      [&](auto const& x) {
        return s(x, t);
      },
      [&](auto const& x) {
        return s.in_domain(x, t);
      },
      min, max, distance_on_ray, std::forward<ColorScale>(color_scale),
      std::forward<AlphaScale>(alpha_scale), bg_color);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <real_number CameraReal, typename Grid, typename Container,
          template <typename> typename Kernel0,
          template <typename> typename Kernel1,
          template <typename> typename Kernel2,
          regular_invocable<typename Container::value_type> ColorScale,
          regular_invocable<typename Container::value_type> AlphaScale>
auto direct_volume_rendering(
    camera<CameraReal> const& cam,
    grid_vertex_property<Grid, Container, Kernel0, Kernel1, Kernel2> const&
                                         prop,
    typename Container::value_type const min,
    typename Container::value_type const max,
    typename Container::value_type const distance_on_ray,
    ColorScale&& color_scale, AlphaScale&& alpha_scale,
    std::invoke_result_t<ColorScale, typename Container::value_type> const&
        bg_color = {}) {
  return direct_volume_rendering(
      cam, prop.grid().boundingbox(),
      [&](auto const& x) {
        return prop.sample(x);
      },
      [](auto const&) {
        return true;
      },
      min, max, distance_on_ray, std::forward<ColorScale>(color_scale),
      std::forward<AlphaScale>(alpha_scale), bg_color);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif