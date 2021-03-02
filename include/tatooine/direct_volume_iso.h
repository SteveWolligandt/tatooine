#ifndef TATOOINE_DIRECT_VOLUME_RENDERING_H
#define TATOOINE_DIRECT_VOLUME_RENDERING_H
//==============================================================================
#include <omp.h>
#include <tatooine/demangling.h>
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/rendering/camera.h>
//==============================================================================
namespace tatooine {
//==============================================================================
#ifdef __cpp_concepts
template <
    arithmetic Min, arithmetic Max, arithmetic DistOnRay, arithmetic CameraReal,
    arithmetic AABBReal, regular_invocable<vec<AABBReal, 3>> DataEvaluator,
    regular_invocable<vec<AABBReal, 3>> MappedDataEvaluator,
    arithmetic Isovalue, regular_invocable<vec<AABBReal, 3>> DomainCheck,
    regular_invocable<std::invoke_result_t<DataEvaluator, vec<AABBReal, 3>>>
        ColorScale,
    regular_invocable<std::invoke_result_t<DataEvaluator, vec<AABBReal, 3>>>
        AlphaScale>
#else
template <typename Min, typename Max, typename DistOnRay, typename CameraReal,
          typename AABBReal, typename DataEvaluator,
          typename MappedDataEvaluator, typename Isovalue, typename DomainCheck,
          typename ColorScale, typename AlphaScale
          //    , enable_if<
          //      is_arithmetic<Min, Max, CameraReal,  AABBReal, DistOnRay>,
          //      is_invocable<DataEvaluator, vec<AABBReal, 3>>,
          //      is_invocable<DomainCheck, vec<AABBReal, 3>>
          //      is_invocable<ColorScale, std::invoke_result_t<DataEvaluator,
          //      vec<AABBReal, 3>>>, is_invocable<AlphaScale,
          //      std::invoke_result_t<DataEvaluator, vec<AABBReal, 3>>>
          //    > = true
          >
#endif
auto direct_volume_iso(
    rendering::camera<CameraReal> const&          cam,
    axis_aligned_bounding_box<AABBReal, 3> const& aabb,
    DataEvaluator&& data_evaluator, Isovalue isovalue,
    MappedDataEvaluator&& mapped_data_evaluator, Min const min, Max const max,
    DomainCheck&& domain_check, DistOnRay const distance_on_ray,
    ColorScale&& color_scale,
    AlphaScale&& alpha_scale,
    std::invoke_result_t<
        ColorScale,
        std::invoke_result_t<DataEvaluator, vec<AABBReal, 3>>> const& bg_color =
        {}) {
  using pos_t      = vec<AABBReal, 3>;
  using value_type = std::invoke_result_t<DataEvaluator, pos_t>;
  using color_type = std::invoke_result_t<ColorScale, value_type>;
  using alpha_type = std::invoke_result_t<AlphaScale, value_type>;
  static_assert(is_arithmetic<value_type>,
                "DataEvaluator must return scalar type.");
  static_assert(is_arithmetic<color_type> || is_vec<color_type>,
                "ColorScale must return scalar type or tatooine::vec.");
  static_assert(is_floating_point<alpha_type>,
                "AlphaScale must return floating point number.");
  grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, cam.plane_width() - 1, cam.plane_width()},
      linspace<CameraReal>{0.0, cam.plane_height() - 1, cam.plane_height()}};
  auto& rendering =
      rendered_image.template add_vertex_property<color_type>("rendering");

  std::vector<std::tuple<ray<CameraReal, 3>, AABBReal, size_t, size_t>> rays;
  std::mutex mutex;
#pragma omp parallel for collapse(2)
  for (size_t y = 0; y < cam.plane_height(); ++y) {
    for (size_t x = 0; x < cam.plane_width(); ++x) {
      rendering(x, y) = bg_color;
      auto r          = cam.ray(x, y);
      r.normalize();
      if (auto const i = aabb.check_intersection(r); i) {
        std::lock_guard lock{mutex};
        rays.push_back(std::tuple{r, i->t, x, y});
      }
    }
  }
#pragma omp parallel for
  for (size_t i = 0; i < rays.size(); ++i) {
    auto const [r, t, x, y]      = rays[i];
    alpha_type accumulated_alpha = 0;
    color_type accumulated_color{};
    auto       t0      = t;
    auto       t1      = t + distance_on_ray;
    pos_t      x0 = r(t0);
    pos_t      x1 = r(t1);
    auto sample0 = data_evaluator(x0);
    auto sample1 = data_evaluator(x1);

    //if (!aabb.is_inside(x0) || !aabb.is_inside(x1)) {
    //  t0 += 1e-6;
    //  t1 += 1e-6;
    //  x0 = r(t0);
    //  x1 = r(t1);
    //  sample0 = data_evaluator(x0);
    //  sample1 = data_evaluator(x1);
    //}
    //
    while (aabb.is_inside(x1) &&
           accumulated_alpha < 0.95) {
      if (domain_check(x0) && domain_check(x1)) {
        if ((sample0 <= isovalue && sample1 > isovalue) ||
            (sample0 >= isovalue && sample1 < isovalue)) {
          auto const t_iso = (isovalue - sample0) / (sample1 - sample0);
          auto const iso_pos = r(t0 + t_iso * distance_on_ray);
          using pos_t = typename decltype(r)::pos_t;

          auto const sample_at_iso = mapped_data_evaluator(iso_pos);
          constexpr auto eps             = 1e-5;
          constexpr auto      offset_x        = pos_t{eps, 0, 0};
          auto                x_fw            = iso_pos + offset_x;
          auto                x_bw            = iso_pos - offset_x;
          auto                dx              = 2 * eps;
          if (!domain_check(x_fw)) {
            x_fw = iso_pos;
            dx = eps;
          }
          if (!domain_check(x_bw)) {
            x_bw = iso_pos;
            dx = eps;
          }

          constexpr auto      offset_y        = pos_t{0, eps, 0};
          auto                y_fw            = iso_pos + offset_y;
          auto                y_bw            = iso_pos - offset_y;
          auto                dy              = 2 * eps;
          if (!domain_check(y_fw)) {
            y_fw = iso_pos;
            dy = eps;
          }
          if (!domain_check(y_bw)) {
            y_bw = iso_pos;
            dy = eps;
          }

          constexpr auto      offset_z        = pos_t{0,0,eps};
          auto                z_fw            = iso_pos + offset_z;
          auto                z_bw            = iso_pos - offset_z;
          auto                dz              = 2 * eps;
          if (!domain_check(z_fw)) {
            z_fw = iso_pos;
            dz = eps;
          }
          if (!domain_check(z_bw)) {
            z_bw = iso_pos;
            dz = eps;
          }

          auto const gradient_at_iso = normalize(vec{
              (mapped_data_evaluator(x_fw) - mapped_data_evaluator(x_bw)) / dx,
              (mapped_data_evaluator(y_fw) - mapped_data_evaluator(y_bw)) / dy,
              (mapped_data_evaluator(z_fw) - mapped_data_evaluator(z_bw)) / dz,
          });
          auto const normalized_sample =
              std::clamp<value_type>((sample_at_iso - min) / (max - min), 0, 1);

          auto const diffuse = 
                  std::abs(dot(r.direction(), gradient_at_iso));
          auto const reflect_dir = reflect(-r.direction(), gradient_at_iso);
          auto const spec_dot = std::max(dot(reflect_dir, r.direction()), 0.0);
          auto const specular = 
                  std::pow(spec_dot, 100) ;
          auto const sample_color =
              color_scale(normalized_sample) * diffuse + specular;

          auto const sample_alpha =
              std::clamp<alpha_type>(alpha_scale(normalized_sample), 0, 1) +
              specular;

          accumulated_color +=
              (1 - accumulated_alpha) * sample_alpha * sample_color;
          accumulated_alpha += (1 - accumulated_alpha) * sample_alpha;
        }
      }
      t0 = t1;
      t1 += distance_on_ray;
      x0 = std::move(x1);
      x1      = r(t1);
      sample0 = std::move(sample1);
      sample1 = data_evaluator(x1);
    }
    rendering(x, y) = accumulated_color * accumulated_alpha +
                      bg_color * (1 - accumulated_alpha);
  }
  return rendered_image;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#ifdef __cpp_concepts
//template <arithmetic TReal, arithmetic Min, arithmetic Max,
//          arithmetic DistOnRay, arithmetic CameraReal, arithmetic AABBReal,
//          typename S, typename SReal, regular_invocable<SReal>    ColorScale,
//          regular_invocable<SReal> AlphaScale>
//#else
//template <
//    typename TReal, typename Min, typename Max, typename DistOnRay,
//    typename CameraReal, typename AABBReal, typename S, typename SReal,
//    typename ColorScale, typename AlphaScale,
//    enable_if<is_arithmetic<TReal, Min, Max, DistOnRay, CameraReal, AABBReal>,
//              is_invocable<ColorScale, SReal>,
//              is_invocable<AlphaScale, SReal>> = true>
//#endif
//auto direct_volume_iso(
//    rendering::camera<CameraReal> const&          cam,
//    axis_aligned_bounding_box<AABBReal, 3> const& aabb,
//    scalarfield<S, SReal, 3> const& s, TReal const t, Min const min,
//    Max const max, DistOnRay const distance_on_ray, ColorScale&& color_scale,
//    AlphaScale&&                                   alpha_scale,
//    std::invoke_result_t<ColorScale, SReal> const& bg_color = {}) {
//  return direct_volume_iso(
//      cam, aabb, [&](auto const& x) { return s(x, t); },
//      [&](auto const& x) { return s.in_domain(x, t); }, min, max,
//      distance_on_ray, std::forward<ColorScale>(color_scale),
//      std::forward<AlphaScale>(alpha_scale), bg_color);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#ifdef __cpp_concepts
//template <arithmetic Min, arithmetic Max, arithmetic DistOnRay,
//          arithmetic CameraReal, typename Grid, typename ValueType,
//          bool HasNonConstReference, regular_invocable<double> ColorScale,
//          regular_invocable<double> AlphaScale>
//#else
//template <typename Min, typename Max, typename DistOnRay, typename CameraReal,
//          typename Grid, typename ValueType, bool HasNonConstReference,
//          typename ColorScale, typename AlphaScale,
//          enable_if<is_arithmetic<Min, Max, DistOnRay, CameraReal>,
//                    is_invocable<ColorScale, double>,
//                    is_invocable<AlphaScale, double>> = true>
//#endif
//auto direct_volume_iso(
//    rendering::camera<CameraReal> const&                                  cam,
//    typed_multidim_property<Grid, ValueType, HasNonConstReference> const& prop,
//    Min const min, Max const max, DistOnRay const distance_on_ray,
//    ColorScale&& color_scale, AlphaScale&& alpha_scale,
//    std::invoke_result_t<ColorScale, ValueType> const& bg_color = {}) {
//  auto sampler = prop.template sampler<interpolation::cubic>();
//  return direct_volume_iso(
//      cam, prop.grid().bounding_box(),
//      [&](auto const& x) { return sampler(x); },
//      [](auto const&) { return true; }, min, max, distance_on_ray,
//      std::forward<ColorScale>(color_scale),
//      std::forward<AlphaScale>(alpha_scale), bg_color);
//}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
