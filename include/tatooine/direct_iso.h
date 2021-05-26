#ifndef TATOOINE_DIRECT_ISO_H
#define TATOOINE_DIRECT_ISO_H
//==============================================================================
#include <omp.h>
#include <tatooine/demangling.h>
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/rendering/camera.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename CameraReal, typename IsoReal, typename GridVertexProperty>
auto direct_iso(
    rendering::camera<CameraReal> const&  cam,
    sampler<GridVertexProperty, interpolation::linear, interpolation::linear,
            interpolation::linear> const& linear_field,
    IsoReal const                         isovalue) {
  auto const& g    = linear_field.grid();
  auto const& dim0 = g.template dimension<0>();
  auto const& dim1 = g.template dimension<1>();
  auto const& dim2 = g.template dimension<2>();
  auto const  aabb = g.bounding_box();
  grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, cam.plane_width() - 1, cam.plane_width()},
      linspace<CameraReal>{0.0, cam.plane_height() - 1, cam.plane_height()}};
  auto& rendering =
      rendered_image.template add_vertex_property<vec3>("rendering");

  std::vector<std::tuple<ray<CameraReal, 3>, double, size_t, size_t>> rays;
  std::mutex mutex;
#pragma omp parallel for collapse(2)
  for (size_t y = 0; y < cam.plane_height(); ++y) {
    for (size_t x = 0; x < cam.plane_width(); ++x) {
      rendering(x, y) = vec3::ones();
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
    auto const [r, t, x, y] = rays[i];

    auto entry_point = r(t);
    for (size_t i = 0; i < 3; ++i) {
      if (entry_point(i) < aabb.min(i)) {
        entry_point(i) = aabb.min(i);
      }
      if (entry_point(i) > aabb.max(i)) {
        entry_point(i) = aabb.max(i);
      }
    }
    auto exit_point = entry_point;

    std::array<std::pair<size_t, double>, 3> cur_indices;
    for (size_t i = 0; i < 3; ++i) {
      cur_indices[i] = g.cell_index(i, entry_point(i));
      if (std::abs(cur_indices[i].second) < 1e-9) {
        cur_indices[i].second = 0;
      } else if (std::abs(cur_indices[i].second - 1) < 1e-9) {
        cur_indices[i].second = 1;
      }
    }

    auto done = false;
    while (!done) {
      auto const next_indices = std::array<size_t, 3>{
          cur_indices[0].first + size_t(cur_indices[0].second) +
              (r.direction(0) > 0 ? 1 : -1),
          cur_indices[1].first + size_t(cur_indices[1].second) +
              (r.direction(1) > 0 ? 1 : -1),
          cur_indices[2].first + size_t(cur_indices[2].second) +
              (r.direction(2) > 0 ? 1 : 0)};
      assert(next_indices[0] < dim0.size() && next_indices[0] != size_t(0) - 1);
      assert(next_indices[1] < dim1.size() && next_indices[1] != size_t(0) - 1);
      assert(next_indices[2] < dim2.size() && next_indices[2] != size_t(0) - 1);
      auto const t0 = (dim0[next_indices[0]] - r.origin(0)) / r.direction(0);
      auto const t1 = (dim1[next_indices[1]] - r.origin(1)) / r.direction(1);
      auto const t2 = (dim2[next_indices[2]] - r.origin(2)) / r.direction(2);
      auto const exit_time = tatooine::min(t0, t1, t2);
      exit_point           = r(exit_time);

      auto const xa    = r.origin(0);
      auto const xb    = r.direction(0);
      auto const x0    = cur_indices[0].first;
      auto const x1    = x0 + 1;
      auto const xnorm = 1 / (x1 - x0);
      auto const ya    = r.origin(1);
      auto const yb    = r.direction(1);
      auto const y0    = cur_indices[1].first;
      auto const y1    = y0 + 1;
      auto const ynorm = 1 / (y1 - y0);
      auto const za    = r.origin(2);
      auto const zb    = r.direction(2);
      auto const z0    = cur_indices[2].first;
      auto const z1    = z0 + 1;
      auto const znorm = 1 / (z1 - z0);
      auto const a0    = vec3{
          (x1 - xa) * xnorm,
          (y1 - ya) * ynorm,
          (z1 - za) * znorm,
      };
      auto const b0 = vec3{
          xb * xnorm,
          yb * ynorm,
          zb * znorm,
      };
      auto const a1 = vec3{
          (xa - x0) * xnorm,
          (ya - y0) * ynorm,
          (za - z0) * znorm,
      };
      auto const b1 = vec3{
          -xb * xnorm,
          -yb * ynorm,
          -zb * znorm,
      };

      auto const A = -isovalue +
                     a0(0) * a0(1) * a0(2) * linear_field.data_at(x0, y0, z0) +
                     a0(0) * a0(1) * a1(2) * linear_field.data_at(x0, y0, z1) +
                     a0(0) * a1(1) * a0(2) * linear_field.data_at(x0, y1, z0) +
                     a0(0) * a1(1) * a1(2) * linear_field.data_at(x0, y1, z1) +
                     a1(0) * a0(1) * a0(2) * linear_field.data_at(x1, y0, z0) +
                     a1(0) * a0(1) * a1(2) * linear_field.data_at(x1, y0, z1) +
                     a1(0) * a1(1) * a0(2) * linear_field.data_at(x1, y1, z0) +
                     a1(0) * a1(1) * a1(2) * linear_field.data_at(x1, y1, z1);
      auto const B =
        (b0(0) * a0(1) * a0(2) + a0(0) * b0(1) * a0(2) + a0(0) * a0(1) * b0(2)) * linear_field.data_at(x0, y0, z0) +
        (b0(0) * a0(1) * a1(2) + a0(0) * b0(1) * a1(2) + a0(0) * a0(1) * b1(2)) * linear_field.data_at(x0, y0, z1) +
        (b0(0) * a1(1) * a0(2) + a0(0) * b1(1) * a0(2) + a0(0) * a1(1) * b0(2)) * linear_field.data_at(x0, y1, z0) +
        (b0(0) * a1(1) * a1(2) + a0(0) * b1(1) * a1(2) + a0(0) * a1(1) * b1(2)) * linear_field.data_at(x0, y1, z1) +
        (b1(0) * a0(1) * a0(2) + a1(0) * b0(1) * a0(2) + a1(0) * a0(1) * b0(2)) * linear_field.data_at(x1, y0, z0) +
        (b1(0) * a0(1) * a1(2) + a1(0) * b0(1) * a1(2) + a1(0) * a0(1) * b1(2)) * linear_field.data_at(x1, y0, z1) +
        (b1(0) * a1(1) * a0(2) + a1(0) * b1(1) * a0(2) + a1(0) * a1(1) * b0(2)) * linear_field.data_at(x1, y1, z0) +
        (b1(0) * a1(1) * a1(2) + a1(0) * b1(1) * a1(2) + a1(0) * a1(1) * b1(2)) * linear_field.data_at(x1, y1, z1);
      auto const C =
        (a0(0) * b0(1) * b0(2) + b0(0) * a0(1) * b0(2) + b0(0) * b0(1) * a0(2)) * linear_field.data_at(x0, y0, z0) +
        (a0(0) * b0(1) * b1(2) + b0(0) * a0(1) * b1(2) + b0(0) * b0(1) * a1(2)) * linear_field.data_at(x0, y0, z1) +
        (a0(0) * b1(1) * b0(2) + b0(0) * a1(1) * b0(2) + b0(0) * b1(1) * a0(2)) * linear_field.data_at(x0, y1, z0) +
        (a0(0) * b1(1) * b1(2) + b0(0) * a1(1) * b1(2) + b0(0) * b1(1) * a1(2)) * linear_field.data_at(x0, y1, z1) +
        (a1(0) * b0(1) * b0(2) + b1(0) * a0(1) * b0(2) + b1(0) * b0(1) * a0(2)) * linear_field.data_at(x1, y0, z0) +
        (a1(0) * b0(1) * b1(2) + b1(0) * a0(1) * b1(2) + b1(0) * b0(1) * a1(2)) * linear_field.data_at(x1, y0, z1) +
        (a1(0) * b1(1) * b0(2) + b1(0) * a1(1) * b0(2) + b1(0) * b1(1) * a0(2)) * linear_field.data_at(x1, y1, z0) +
        (a1(0) * b1(1) * b1(2) + b1(0) * a1(1) * b1(2) + b1(0) * b1(1) * a1(2)) * linear_field.data_at(x1, y1, z1);
      auto const D = b0(0) * b0(1) * b0(2) * linear_field.data_at(x0, y0, z0) +
                     b0(0) * b0(1) * b1(2) * linear_field.data_at(x0, y0, z1) +
                     b0(0) * b1(1) * b0(2) * linear_field.data_at(x0, y1, z0) +
                     b0(0) * b1(1) * b1(2) * linear_field.data_at(x0, y1, z1) +
                     b1(0) * b0(1) * b0(2) * linear_field.data_at(x1, y0, z0) +
                     b1(0) * b0(1) * b1(2) * linear_field.data_at(x1, y0, z1) +
                     b1(0) * b1(1) * b0(2) * linear_field.data_at(x1, y1, z0) +
                     b1(0) * b1(1) * b1(2) * linear_field.data_at(x1, y1, z1);

      auto const s = solve(polynomial{A, B, C, D});
      if (!s.empty()) {
        for (auto const t : s) {
           auto const c = a0 + t*b0;
           if (c(0) >= 0 && c(0) <= 1 &&
               c(1) >= 0 && c(1) <= 1 &&
               c(2) >= 0 && c(2) <= 1) {
             rendering(x, y) = vec3{1, 0, 0};
           }
        }
      }

      entry_point = std::move(exit_point);
      for (size_t i = 0; i < 3; ++i) {
        cur_indices[i] = g.cell_index(i, entry_point(i));
        if (std::abs(cur_indices[i].second) < 1e-9) {
          cur_indices[i].second = 0;
        } else if (std::abs(cur_indices[i].second - 1) < 1e-9) {
          cur_indices[i].second = 1;
        }
        if ((r.direction(i) < 0 &&
             cur_indices[i].first + size_t(cur_indices[i].second) == 0) ||
            (r.direction(i) > 0 &&
             cur_indices[i].first + size_t(cur_indices[i].second) ==
                 g.size(i) - 1)) {
          done = true;
          break;
        }
      }
    }
  }
  rendering.write_png("das.png");
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <
    arithmetic Min, arithmetic Max, arithmetic DistOnRay, arithmetic CameraReal,
    arithmetic AABBReal, regular_invocable<vec<AABBReal, 3>> DataEvaluator,
    regular_invocable<vec<AABBReal, 3>> GradientDataEvaluator,
    regular_invocable<vec<AABBReal, 3>> MappedDataEvaluator,
    arithmetic Isovalue, regular_invocable<vec<AABBReal, 3>> DomainCheck,
    regular_invocable<std::invoke_result_t<DataEvaluator, vec<AABBReal, 3>>>
        ColorScale,
    regular_invocable<std::invoke_result_t<DataEvaluator, vec<AABBReal, 3>>>
        AlphaScale>
#else
template <typename Min, typename Max, typename DistOnRay, typename CameraReal,
          typename AABBReal, typename DataEvaluator,
          typename GradientDataEvaluator, typename MappedDataEvaluator,
          typename Isovalue, typename DomainCheck, typename ColorScale,
          typename AlphaScale>
#endif
auto direct_iso(
    rendering::camera<CameraReal> const&          cam,
    axis_aligned_bounding_box<AABBReal, 3> const& aabb,
    DataEvaluator&& data_evaluator, GradientDataEvaluator&& gradient_data_evaluator,
    Isovalue isovalue, MappedDataEvaluator&& mapped_data_evaluator,
    Min const min, Max const max, DomainCheck&& domain_check,
    DistOnRay const distance_on_ray, ColorScale&& color_scale,
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

    auto t0      = t;
    auto x0      = r(t0);
    for (size_t i = 0; i < 3; ++i) {
      if (x0(i) < aabb.min(i)) {
        x0(i) = aabb.min(i);
      }
      if (x0(i) > aabb.max(i)) {
        x0(i) = aabb.max(i);
      }
    }
    auto sample0 = data_evaluator(x0);

    auto  t1      = t0;
    auto  x1      = x0;
    auto  sample1 = sample0;

    //if (!aabb.is_inside(x0) || !aabb.is_inside(x1)) {
    //  t0 += 1e-6;
    //  t1 += 1e-6;
    //  x0 = r(t0);
    //  x1 = r(t1);
    //  sample0 = data_evaluator(x0);
    //  sample1 = data_evaluator(x1);
    //}
    //
    while (domain_check(x0) && accumulated_alpha < 0.95) {
      t1 += distance_on_ray;
      x1 = r(t1);
      if (domain_check(x1)) {
        sample1 = data_evaluator(x1);
        if ((sample0 <= isovalue && sample1 > isovalue) ||
            (sample0 >= isovalue && sample1 < isovalue)) {
          auto cur_x0 = x0;
          auto cur_x1 = x1;
          auto cur_sample0 = sample0;
          auto cur_sample1 = sample1;
          for (size_t i = 0; i < 10; ++i) {
            auto x_center = (cur_x0 + cur_x1) / 2;
            auto sample_center = data_evaluator(x_center);
            if ((cur_sample0 <= isovalue && sample_center > isovalue) ||
                (cur_sample0 >= isovalue && sample_center < isovalue)) {
              cur_x1 = x_center;
              cur_sample1 = sample_center;
            } else {
              cur_x0 = x_center;
              cur_sample0 = sample_center;
            }
          }
          auto const t_iso = (isovalue - cur_sample0) / (cur_sample1 - cur_sample0);
          auto const iso_pos = r(t0 + t_iso * distance_on_ray);

          auto const sample_at_iso = mapped_data_evaluator(iso_pos);
          auto const normalized_sample =
              std::clamp<value_type>((sample_at_iso - min) / (max - min), 0, 1);

          auto const gradient_at_iso = gradient_data_evaluator(iso_pos);
          auto const normal          = normalize(gradient_at_iso);
          auto const diffuse = std::abs(dot(r.direction(), normal));
          auto const reflect_dir     = reflect(-r.direction(), normal);
          auto const spec_dot = std::max(dot(reflect_dir, r.direction()), 0.0);
          auto const specular = std::pow(spec_dot, 100);
          auto const sample_color =
            //iso_pos * 0.5+0.5;
              //color_scale(normalized_sample) * diffuse + specular;
              //color_scale(normalized_sample) * diffuse ;
              color_scale(normalized_sample);

          auto const sample_alpha =
              std::clamp<alpha_type>(alpha_scale(normalized_sample), 0, 1)
              //+ specular
              ;

          accumulated_color +=
              (1 - accumulated_alpha) * sample_alpha * sample_color;
          accumulated_alpha += (1 - accumulated_alpha) * sample_alpha;
        }
      }
      t0 = t1;
      x0 = std::move(x1);
      sample0 = std::move(sample1);
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
//auto direct_iso(
//    rendering::camera<CameraReal> const&          cam,
//    axis_aligned_bounding_box<AABBReal, 3> const& aabb,
//    scalarfield<S, SReal, 3> const& s, TReal const t, Min const min,
//    Max const max, DistOnRay const distance_on_ray, ColorScale&& color_scale,
//    AlphaScale&&                                   alpha_scale,
//    std::invoke_result_t<ColorScale, SReal> const& bg_color = {}) {
//  return direct_iso(
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
//auto direct_iso(
//    rendering::camera<CameraReal> const&                                  cam,
//    typed_multidim_property<Grid, ValueType, HasNonConstReference> const& prop,
//    Min const min, Max const max, DistOnRay const distance_on_ray,
//    ColorScale&& color_scale, AlphaScale&& alpha_scale,
//    std::invoke_result_t<ColorScale, ValueType> const& bg_color = {}) {
//  auto sampler = prop.template sampler<interpolation::cubic>();
//  return direct_iso(
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
