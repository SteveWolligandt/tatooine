#ifndef TATOOINE_RENDERING_DIRECT_VOLUME_H
#define TATOOINE_RENDERING_DIRECT_VOLUME_H
//==============================================================================
#include <omp.h>
#include <tatooine/demangling.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/camera.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
template <arithmetic DistOnRay, arithmetic CameraReal, arithmetic AABBReal,
          regular_invocable<vec<AABBReal, 3>> DomainCheck, typename Shader>
auto direct_volume(camera<CameraReal> const&                     cam,
                   axis_aligned_bounding_box<AABBReal, 3> const& aabb,
                   DomainCheck&& domain_check, DistOnRay const distance_on_ray,
                   Shader&& shader) {
  using pos_t     = vec<AABBReal, 3>;
  using viewdir_t = vec<CameraReal, 3>;
  using color_t   = std::invoke_result_t<Shader, pos_t, viewdir_t>;
  using rgb_t     = vec<typename color_t::value_type, 3>;
  using alpha_t   = typename color_t::value_type;
  static_assert(is_vec<color_t> && color_t::num_components() == 4,
                "Shader must return a vector with 4 components.");
  rectilinear_grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, cam.plane_width() - 1, cam.plane_width()},
      linspace<CameraReal>{0.0, cam.plane_height() - 1, cam.plane_height()}};
  auto& rendering =
      rendered_image.template insert_vertex_property<rgb_t>("rendering");
  auto const bg_color = rgb_t::ones();

  std::vector<std::tuple<ray<CameraReal, 3>, AABBReal, size_t, size_t>> rays;
  for (size_t y = 0; y < cam.plane_height(); ++y) {
    for (size_t x = 0; x < cam.plane_width(); ++x) {
      rendering(x, y) = bg_color;
      auto r          = cam.ray(x, y);
      r.normalize();
      if (auto const i = aabb.check_intersection(r); i) {
        rays.push_back(std::tuple{r, i->t, x, y});
      }
    }
  }
#pragma omp parallel for
  for (size_t i = 0; i < rays.size(); ++i) {
    auto const [r, t, x, y] = rays[i];
    auto  accumulated_color = rgb_t::zeros();
    auto  accumulated_alpha = alpha_t(0);
    auto  cur_t             = t;
    pos_t cur_pos           = r(cur_t);
    if (!aabb.is_inside(cur_pos)) {
      cur_t += 1e-6;
      cur_pos = r(cur_t);
    }

    while (aabb.is_inside(cur_pos) && accumulated_alpha < 0.95) {
      if (domain_check(cur_pos)) {
        auto const rgba  = shader(cur_pos, r.direction());
        auto const rgb   = vec{rgba(0), rgba(1), rgba(2)};
        auto const alpha = rgba(3);
        accumulated_color += (1 - accumulated_alpha) * alpha * rgb;
        accumulated_alpha += (1 - accumulated_alpha) * alpha;
      }
      cur_t += distance_on_ray;
      cur_pos = r(cur_t);
    }
    rendering(x, y) = accumulated_color * accumulated_alpha +
                      bg_color * (1 - accumulated_alpha);
  }
  return rendered_image;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// template <arithmetic TReal, arithmetic Min, arithmetic Max,
//          arithmetic DistOnRay, arithmetic CameraReal, arithmetic AABBReal,
//          typename S, typename SReal, regular_invocable<SReal>    ColorScale,
//          regular_invocable<SReal> AlphaScale>
// auto direct_volume(
//    camera<CameraReal> const&                     cam,
//    axis_aligned_bounding_box<AABBReal, 3> const& aabb,
//    scalarfield<S, SReal, 3> const& s, TReal const t, Min const min,
//    Max const max, DistOnRay const distance_on_ray, ColorScale&& color_scale,
//    AlphaScale&&                                   alpha_scale,
//    std::invoke_result_t<ColorScale, SReal> const& bg_color = {}) {
//  return direct_volume(
//      cam, aabb, [&](auto const& x) { return s(x, t); },
//      [&](auto const& x) { return s.in_domain(x, t); }, min, max,
//      distance_on_ray, std::forward<ColorScale>(color_scale),
//      std::forward<AlphaScale>(alpha_scale), bg_color);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
// template <arithmetic Min, arithmetic Max, arithmetic DistOnRay,
//          arithmetic CameraReal, typename Grid, typename ValueType,
//          bool HasNonConstReference, regular_invocable<double> ColorScale,
//          regular_invocable<double> AlphaScale>
// auto direct_volume(
//    camera<CameraReal> const&                                         cam,
//    typed_grid_vertex_property_interface<Grid, ValueType,
//                                         HasNonConstReference> const& prop,
//    Min const min, Max const max, DistOnRay const distance_on_ray,
//    ColorScale&& color_scale, AlphaScale&& alpha_scale,
//    std::invoke_result_t<ColorScale, ValueType> const& bg_color = {}) {
//  auto sampler = prop.template sampler<interpolation::cubic>();
//  return direct_volume(
//      cam, prop.grid().bounding_box(),
//      [&](auto const& x) { return sampler(x); },
//      [](auto const&) { return true; }, min, max, distance_on_ray,
//      std::forward<ColorScale>(color_scale),
//      std::forward<AlphaScale>(alpha_scale), bg_color);
//}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
