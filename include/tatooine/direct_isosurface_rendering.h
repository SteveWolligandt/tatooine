/// \file direct_isosurface_rendering.h
/// This file specifies functions for direct renderings of iso surfaces.
#ifndef TATOOINE_DIRECT_ISOSURFACE_RENDERING_H
#define TATOOINE_DIRECT_ISOSURFACE_RENDERING_H
//==============================================================================
#include <omp.h>
#include <tatooine/demangling.h>
#include <tatooine/field.h>
#include <tatooine/grid.h>
#include <tatooine/rendering/camera.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// This is an implementation of \cite 10.5555/288216.288266. See also \ref
/// isosurf_parker.
///
/// \param cam Camera model for casting rays
/// \param linear_field Piece-wise trilinear field
/// \param isovalue Iso Value of the extracted iso surface
/// \param shader Shader for setting color at pixel. The shader takes position,
/// color (vec3 or vec4). \return Returns a 2D grid with a grid_vertex_property
/// named "rendered_isosurface"
template <typename CameraReal, typename IsoReal, typename Dim0, typename Dim1,
          typename Dim2, typename Field, typename Shader>
auto direct_isosurface_rendering(rendering::camera<CameraReal> const& cam,
                                 grid<Dim0, Dim1, Dim2> const& g, Field&& field,
                                 IsoReal const isovalue, Shader&& shader) {
  using grid_real_t = typename grid<Dim0, Dim1, Dim2>::real_t;
  using pos_t       = vec<grid_real_t, 3>;
  constexpr auto use_indices =
      std::is_invocable_v<Field, size_t, size_t, size_t>;
  // using value_t =
  //    std::conditional_t<use_indices,
  //                       std::invoke_result_t<Field, size_t, size_t, size_t>,
  //                       std::invoke_result_t<Field, pos_t>>;
  // static_assert(is_floating_point<value_t>);
  using viewdir_t = vec<CameraReal, 3>;
  static_assert(std::is_invocable_v<Shader, pos_t, pos_t, viewdir_t>,
                "Shader must be invocable with position, gradient and view direction.");
  using color_t = std::invoke_result_t<Shader, pos_t, pos_t, viewdir_t>;
  using rgb_t   = vec<typename color_t::value_type, 3>;
  using alpha_t = typename color_t::value_type;
  static_assert(is_vec<color_t>,
                "Shader must return a vector with 3 or 4 components.");
  static_assert(
      color_t::num_components() == 3 || color_t::num_components() == 4,
      "Shader must return a vector with 3 or 4 components.");
  auto const& dim0 = g.template dimension<0>();
  auto const& dim1 = g.template dimension<1>();
  auto const& dim2 = g.template dimension<2>();
  auto const  aabb = g.bounding_box();
  grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, cam.plane_width() - 1, cam.plane_width()},
      linspace<CameraReal>{0.0, cam.plane_height() - 1, cam.plane_height()}};
  auto& rendering =
      rendered_image.template vertex_property<rgb_t>("rendered_isosurface");

  std::vector<std::tuple<ray<CameraReal, 3>, double, size_t, size_t>> rays;
  std::mutex                                                          mutex;
  auto const bg_color = rgb_t{1, 1, 1};
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
    auto cell_data         = std::make_unique<double[]>(8);
    auto indexing          = static_multidim_size<x_fastest, 2, 2, 2>{};
    auto accumulated_color = rgb_t::zeros();
    if constexpr (color_t::num_components() == 3) {
      accumulated_color = bg_color;
    }
    auto accumulated_alpha  = alpha_t(0);
    auto const& [r, t, x, y] = rays[i];

    auto entry_point = r(t);
    for (size_t i = 0; i < 3; ++i) {
      if (entry_point(i) < aabb.min(i)) {
        entry_point(i) = aabb.min(i);
      }
      if (entry_point(i) > aabb.max(i)) {
        entry_point(i) = aabb.max(i);
      }
    }

    auto cell_pos        = make_array<real_t, 3>();
    auto done            = false;
    auto update_cell_pos = [&](auto const& r) {
      for (size_t dim = 0; dim < 3; ++dim) {
        auto const [ci, t] = g.cell_index(dim, entry_point(dim));
        if (std::abs(t) < 1e-7) {
          cell_pos[dim] = ci;
          if (r.direction(dim) < 0 && ci == 0) {
            done = true;
          }
        } else if (std::abs(t - 1) < 1e-7) {
          cell_pos[dim] = ci + 1;
          if (r.direction(dim) > 0 && ci + 1 == g.size(dim) - 1) {
            done = true;
          }
        } else {
          cell_pos[dim] = ci + t;
        }
        assert(cell_pos[dim] >= 0 && cell_pos[dim] <= g.size(dim) - 1);
      }
    };
    update_cell_pos(r);

    while (!done) {
      auto plane_indices_to_check = make_array<size_t, 3>();
      for (size_t dim = 0; dim < 3; ++dim) {
        if (cell_pos[dim] - std::floor(cell_pos[dim]) == 0) {
          if (r.direction(dim) > 0) {
            plane_indices_to_check[dim] =
                static_cast<size_t>(cell_pos[dim]) + 1;
          } else {
            plane_indices_to_check[dim] =
                static_cast<size_t>(cell_pos[dim]) - 1;
          }
        } else {
          if (r.direction(dim) > 0) {
            plane_indices_to_check[dim] = std::ceil(cell_pos[dim]);
          } else {
            plane_indices_to_check[dim] = std::floor(cell_pos[dim]);
          }
        }
      }

      assert(plane_indices_to_check[0] < dim0.size() &&
             plane_indices_to_check[0] != size_t(0) - 1);
      assert(plane_indices_to_check[1] < dim1.size() &&
             plane_indices_to_check[1] != size_t(0) - 1);
      assert(plane_indices_to_check[2] < dim2.size() &&
             plane_indices_to_check[2] != size_t(0) - 1);
      auto const t0 = r.direction(0) == 0
                          ? std::numeric_limits<real_t>::max()
                          : (dim0[plane_indices_to_check[0]] - r.origin(0)) /
                                r.direction(0);
      auto const t1 = r.direction(1) == 0
                          ? std::numeric_limits<real_t>::max()
                          : (dim1[plane_indices_to_check[1]] - r.origin(1)) /
                                r.direction(1);
      auto const t2 = r.direction(2) == 0
                          ? std::numeric_limits<real_t>::max()
                          : (dim2[plane_indices_to_check[2]] - r.origin(2)) /
                                r.direction(2);

      std::array<size_t, 3> i0{0, 0, 0};
      std::array<size_t, 3> i1{0, 0, 0};
      for (size_t dim = 0; dim < 3; ++dim) {
        if (cell_pos[dim] - std::floor(cell_pos[dim]) == 0) {
          if (r.direction(dim) > 0) {
            i0[dim] = static_cast<size_t>(cell_pos[dim]);
            i1[dim] = static_cast<size_t>(cell_pos[dim]) + 1;
          } else if (r.direction(dim) < 0) {
            i0[dim] = static_cast<size_t>(cell_pos[dim]) - 1;
            i1[dim] = static_cast<size_t>(cell_pos[dim]);
          }
        } else {
          i0[dim] = static_cast<size_t>(std::floor(cell_pos[dim]));
          i1[dim] = static_cast<size_t>(std::ceil(cell_pos[dim]));
        }
      }
      if constexpr (use_indices) {
        cell_data[indexing(0, 0, 0)] = field(i0[0], i0[1], i0[2]);
        cell_data[indexing(1, 0, 0)] = field(i1[0], i0[1], i0[2]);
        cell_data[indexing(0, 1, 0)] = field(i0[0], i1[1], i0[2]);
        cell_data[indexing(1, 1, 0)] = field(i1[0], i1[1], i0[2]);
        cell_data[indexing(0, 0, 1)] = field(i0[0], i0[1], i1[2]);
        cell_data[indexing(1, 0, 1)] = field(i1[0], i0[1], i1[2]);
        cell_data[indexing(0, 1, 1)] = field(i0[0], i1[1], i1[2]);
        cell_data[indexing(1, 1, 1)] = field(i1[0], i1[1], i1[2]);
      } else {
        cell_data[indexing(0, 0, 0)] = field(g(i0[0], i0[1], i0[2]));
        cell_data[indexing(1, 0, 0)] = field(g(i1[0], i0[1], i0[2]));
        cell_data[indexing(0, 1, 0)] = field(g(i0[0], i1[1], i0[2]));
        cell_data[indexing(1, 1, 0)] = field(g(i1[0], i1[1], i0[2]));
        cell_data[indexing(0, 0, 1)] = field(g(i0[0], i0[1], i1[2]));
        cell_data[indexing(1, 0, 1)] = field(g(i1[0], i0[1], i1[2]));
        cell_data[indexing(0, 1, 1)] = field(g(i0[0], i1[1], i1[2]));
        cell_data[indexing(1, 1, 1)] = field(g(i1[0], i1[1], i1[2]));
      }

      // check if isosurface is present in current cell
      if (!((cell_data[indexing(0, 0, 0)] > isovalue && cell_data[indexing(0, 0, 1)] > isovalue &&
             cell_data[indexing(0, 1, 0)] > isovalue && cell_data[indexing(0, 1, 1)] > isovalue &&
             cell_data[indexing(1, 0, 0)] > isovalue && cell_data[indexing(1, 0, 1)] > isovalue &&
             cell_data[indexing(1, 1, 0)] > isovalue && cell_data[indexing(1, 1, 1)] > isovalue) ||
            (cell_data[indexing(0, 0, 0)] < isovalue && cell_data[indexing(0, 0, 1)] < isovalue &&
             cell_data[indexing(0, 1, 0)] < isovalue && cell_data[indexing(0, 1, 1)] < isovalue &&
             cell_data[indexing(1, 0, 0)] < isovalue && cell_data[indexing(1, 0, 1)] < isovalue &&
             cell_data[indexing(1, 1, 0)] < isovalue && cell_data[indexing(1, 1, 1)] < isovalue))) {
        auto const  x0 = g(i0[0], i0[1], i0[2]);
        auto const  x1 = g(i1[0], i1[1], i1[2]);
        auto const& xa = r.origin();
        auto const& xb = r.direction();

        auto const cell_extent = x1 - x0;
        auto const inv_cell_extent =
            pos_t{1 / cell_extent(0), 1 / cell_extent(1), 1 / cell_extent(2)};
        // create rays in different spaces
        auto const a0 = (x1 - xa) * inv_cell_extent;
        auto const b0 = xb * inv_cell_extent;
        auto const a1 = (xa - x0) * inv_cell_extent;
        auto const b1 = -xb * inv_cell_extent;

        // construct coefficients of cubic polynomial A + B*t + C*t*t + D*t*t*t 
        auto const A = a0(0) * a0(1) * a0(2) * cell_data[indexing(0, 0, 0)] +
                       a0(0) * a0(1) * a1(2) * cell_data[indexing(0, 0, 1)] +
                       a0(0) * a1(1) * a0(2) * cell_data[indexing(0, 1, 0)] +
                       a0(0) * a1(1) * a1(2) * cell_data[indexing(0, 1, 1)] +
                       a1(0) * a0(1) * a0(2) * cell_data[indexing(1, 0, 0)] +
                       a1(0) * a0(1) * a1(2) * cell_data[indexing(1, 0, 1)] +
                       a1(0) * a1(1) * a0(2) * cell_data[indexing(1, 1, 0)] +
                       a1(0) * a1(1) * a1(2) * cell_data[indexing(1, 1, 1)] - isovalue;
        auto const B = (b0(0) * a0(1) * a0(2) + a0(0) * b0(1) * a0(2) +
                        a0(0) * a0(1) * b0(2)) *
                           cell_data[indexing(0, 0, 0)] +
                       (b0(0) * a0(1) * a1(2) + a0(0) * b0(1) * a1(2) +
                        a0(0) * a0(1) * b1(2)) *
                           cell_data[indexing(0, 0, 1)] +
                       (b0(0) * a1(1) * a0(2) + a0(0) * b1(1) * a0(2) +
                        a0(0) * a1(1) * b0(2)) *
                           cell_data[indexing(0, 1, 0)] +
                       (b0(0) * a1(1) * a1(2) + a0(0) * b1(1) * a1(2) +
                        a0(0) * a1(1) * b1(2)) *
                           cell_data[indexing(0, 1, 1)] +
                       (b1(0) * a0(1) * a0(2) + a1(0) * b0(1) * a0(2) +
                        a1(0) * a0(1) * b0(2)) *
                           cell_data[indexing(1, 0, 0)] +
                       (b1(0) * a0(1) * a1(2) + a1(0) * b0(1) * a1(2) +
                        a1(0) * a0(1) * b1(2)) *
                           cell_data[indexing(1, 0, 1)] +
                       (b1(0) * a1(1) * a0(2) + a1(0) * b1(1) * a0(2) +
                        a1(0) * a1(1) * b0(2)) *
                           cell_data[indexing(1, 1, 0)] +
                       (b1(0) * a1(1) * a1(2) + a1(0) * b1(1) * a1(2) +
                        a1(0) * a1(1) * b1(2)) *
                           cell_data[indexing(1, 1, 1)];
        auto const C = (a0(0) * b0(1) * b0(2) + b0(0) * a0(1) * b0(2) +
                        b0(0) * b0(1) * a0(2)) *
                           cell_data[indexing(0, 0, 0)] +
                       (a0(0) * b0(1) * b1(2) + b0(0) * a0(1) * b1(2) +
                        b0(0) * b0(1) * a1(2)) *
                           cell_data[indexing(0, 0, 1)] +
                       (a0(0) * b1(1) * b0(2) + b0(0) * a1(1) * b0(2) +
                        b0(0) * b1(1) * a0(2)) *
                           cell_data[indexing(0, 1, 0)] +
                       (a0(0) * b1(1) * b1(2) + b0(0) * a1(1) * b1(2) +
                        b0(0) * b1(1) * a1(2)) *
                           cell_data[indexing(0, 1, 1)] +
                       (a1(0) * b0(1) * b0(2) + b1(0) * a0(1) * b0(2) +
                        b1(0) * b0(1) * a0(2)) *
                           cell_data[indexing(1, 0, 0)] +
                       (a1(0) * b0(1) * b1(2) + b1(0) * a0(1) * b1(2) +
                        b1(0) * b0(1) * a1(2)) *
                           cell_data[indexing(1, 0, 1)] +
                       (a1(0) * b1(1) * b0(2) + b1(0) * a1(1) * b0(2) +
                        b1(0) * b1(1) * a0(2)) *
                           cell_data[indexing(1, 1, 0)] +
                       (a1(0) * b1(1) * b1(2) + b1(0) * a1(1) * b1(2) +
                        b1(0) * b1(1) * a1(2)) *
                           cell_data[indexing(1, 1, 1)];
        auto const D = b0(0) * b0(1) * b0(2) * cell_data[indexing(0, 0, 0)] +
                       b0(0) * b0(1) * b1(2) * cell_data[indexing(0, 0, 1)] +
                       b0(0) * b1(1) * b0(2) * cell_data[indexing(0, 1, 0)] +
                       b0(0) * b1(1) * b1(2) * cell_data[indexing(0, 1, 1)] +
                       b1(0) * b0(1) * b0(2) * cell_data[indexing(1, 0, 0)] +
                       b1(0) * b0(1) * b1(2) * cell_data[indexing(1, 0, 1)] +
                       b1(0) * b1(1) * b0(2) * cell_data[indexing(1, 1, 0)] +
                       b1(0) * b1(1) * b1(2) * cell_data[indexing(1, 1, 1)];

        auto const s = solve(polynomial{A, B, C, D});
        if (!s.empty()) {
          std::optional<real_t> best_t;
          pos_t                 uvw1;
          for (auto const t : s) {
            if (auto x = a0 + t * b0;
                - 1e-7 <= x(0) && x(0) <= 1 + 1e-7 &&  //
                -1e-7 <= x(1) && x(1) <= 1 + 1e-7 &&   //
                -1e-7 <= x(2) && x(2) <= 1 + 1e-7 &&   //
                t < best_t.value_or(std::numeric_limits<real_t>::max())) {
              best_t = t;
              uvw1   = x;
            }
          }
          if (best_t) {
            auto const uvw0  = pos_t{1 - uvw1(0), 1 - uvw1(1), 1 - uvw1(2)};
            auto const x_iso = uvw0 * cell_extent + x0;
            assert(uvw0(0) >= 0 && uvw0(0) <= 1);
            assert(uvw0(1) >= 0 && uvw0(1) <= 1);
            assert(uvw0(2) >= 0 && uvw0(2) <= 1);
            assert(uvw1(0) >= 0 && uvw1(0) <= 1);
            assert(uvw1(1) >= 0 && uvw1(1) <= 1);
            assert(uvw1(2) >= 0 && uvw1(2) <= 1);
            auto const k = cell_data[indexing(1, 1, 1)] - cell_data[indexing(0, 1, 1)] -
                           cell_data[indexing(1, 0, 1)] + cell_data[indexing(0, 0, 1)] -
                           cell_data[indexing(1, 1, 0)] + cell_data[indexing(0, 1, 0)] +
                           cell_data[indexing(1, 0, 0)] - cell_data[indexing(0, 0, 0)];
            auto const gradient =
                vec{(k * uvw0(1) + cell_data[indexing(1, 0, 1)] - cell_data[indexing(0, 0, 1)] -
                     cell_data[indexing(1, 0, 0)] + cell_data[indexing(0, 0, 0)]) *
                            uvw0(2) +
                        (cell_data[indexing(1, 1, 0)] - cell_data[indexing(0, 1, 0)] -
                         cell_data[indexing(1, 0, 0)] + cell_data[indexing(0, 0, 0)]) *
                            uvw0(1) +
                        cell_data[indexing(1, 0, 0)] - cell_data[indexing(0, 0, 0)],
                    (k * uvw0(0) + cell_data[indexing(0, 1, 1)] - cell_data[indexing(0, 0, 1)] -
                     cell_data[indexing(0, 1, 0)] + cell_data[indexing(0, 0, 0)]) *
                            uvw0(2) +
                        (cell_data[indexing(1, 1, 0)] - cell_data[indexing(0, 1, 0)] -
                         cell_data[indexing(1, 0, 0)] + cell_data[indexing(0, 0, 0)]) *
                            uvw0(0) +
                        cell_data[indexing(0, 1, 0)] - cell_data[indexing(0, 0, 0)],
                    (k * uvw0(0) + cell_data[indexing(0, 1, 1)] - cell_data[indexing(0, 0, 1)] -
                     cell_data[indexing(0, 1, 0)] + cell_data[indexing(0, 0, 0)]) *
                            uvw0(1) +
                        (cell_data[indexing(1, 0, 1)] - cell_data[indexing(0, 0, 1)] -
                         cell_data[indexing(1, 0, 0)] + cell_data[indexing(0, 0, 0)]) *
                            uvw0(0) +
                        cell_data[indexing(0, 0, 1)] - cell_data[indexing(0, 0, 0)]};
            if constexpr (color_t::num_components() == 3) {
              accumulated_color = shader(x_iso, gradient, r.direction());
              done              = true;
            } else if constexpr (color_t::num_components() == 4) {
              auto const rgba  = shader(x_iso, gradient, r.direction());
              auto const rgb   = vec{rgba(0), rgba(1), rgba(2)};
              auto const alpha = rgba(3);
              accumulated_color += (1 - accumulated_alpha) *  alpha * rgb;
              accumulated_alpha += (1 - accumulated_alpha) * alpha;
              if (accumulated_alpha >= 0.95) {
                done = true;
              }
            }
          }
        }
      }

      if (!done) {
        entry_point = r(tatooine::min(t0, t1, t2));
        update_cell_pos(r);
      }
    }
    if constexpr (color_t::num_components() == 3) {
      rendering(x, y) = accumulated_color;
    } else if constexpr (color_t::num_components() == 4) {
      rendering(x, y) = accumulated_color * accumulated_alpha +
                        bg_color * (1 - accumulated_alpha)
                        ;
    }
  }
  return rendered_image;
}
//==============================================================================
/// an implementation of \cite 10.5555/288216.288266. See also \ref
/// isosurf_parker.
///
/// \param cam Camera model for casting rays
/// \param linear_field Piece-wise trilinear field
/// \param isovalue Iso Value of the extracted iso surface
/// \param shader Shader for setting color at pixel. The shader takes position
/// and view direction as parameters. It must return a RGB or a RGBA
/// color (vec3 or vec4). \return Returns a 2D grid with a grid_vertex_property
/// named "rendered_isosurface"
template <typename CameraReal, typename IsoReal, typename GridVertexProperty,
          typename Shader>
auto direct_isosurface_rendering(
    rendering::camera<CameraReal> const&  cam,
    sampler<GridVertexProperty, interpolation::linear, interpolation::linear,
            interpolation::linear> const& linear_field,
    IsoReal const isovalue, Shader&& shader) {
  return direct_isosurface_rendering(
      cam, linear_field.grid(),
      [&](size_t const ix, size_t const iy, size_t const iz) -> auto const& {
        return linear_field.data_at(ix, iy, iz);
      },
      isovalue, std::forward<Shader>(shader));
}
//------------------------------------------------------------------------------
template <typename DistOnRay, typename CameraReal, typename AABBReal,
          typename DataEvaluator, typename Isovalue, typename DomainCheck,
          typename Shader>
auto direct_isosurface_rendering(
    rendering::camera<CameraReal> const&          cam,
    axis_aligned_bounding_box<AABBReal, 3> const& aabb,
    DataEvaluator&& data_evaluator, DomainCheck&& domain_check,
    Isovalue isovalue, DistOnRay const distance_on_ray, Shader&& shader) {
  using pos_t     = vec<CameraReal, 3>;
  using viewdir_t = vec<CameraReal, 3>;
  static_assert(std::is_invocable_v<Shader, pos_t, viewdir_t>,
                "Shader must be invocable with position and view direction.");
  using value_t = std::invoke_result_t<DataEvaluator, pos_t>;
  using color_t = std::invoke_result_t<Shader, pos_t, viewdir_t>;
  using rgb_t   = vec<typename color_t::value_type, 3>;
  using alpha_t = typename color_t::value_type;
  static_assert(is_floating_point<value_t>,
                "DataEvaluator must return scalar type.");
  static_assert(is_vec<color_t>,
                "ColorScale must return scalar type or tatooine::vec.");
  static_assert(
      color_t::num_components() == 3 || color_t::num_components() == 4,
      "ColorScale must return scalar type or tatooine::vec.");
  grid<linspace<CameraReal>, linspace<CameraReal>> rendered_image{
      linspace<CameraReal>{0.0, cam.plane_width() - 1, cam.plane_width()},
      linspace<CameraReal>{0.0, cam.plane_height() - 1, cam.plane_height()}};
  auto& rendering =
      rendered_image.template vertex_property<rgb_t>("rendered_isosurface");

  std::vector<std::tuple<ray<CameraReal, 3>, AABBReal, size_t, size_t>> rays;
  std::mutex                                                            mutex;
  auto const bg_color = rgb_t{1, 1, 1};
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
    auto const [r, t, x, y]   = rays[i];
    alpha_t accumulated_alpha = 0;
    rgb_t   accumulated_color{};

    auto t0 = t;
    auto x0 = r(t0);
    for (size_t i = 0; i < 3; ++i) {
      if (x0(i) < aabb.min(i)) {
        x0(i) = aabb.min(i);
      }
      if (x0(i) > aabb.max(i)) {
        x0(i) = aabb.max(i);
      }
    }
    auto sample0 = data_evaluator(x0);

    auto t1      = t0 + distance_on_ray;
    auto x1      = r(t1);
    auto sample1 = sample0;

    auto done = false;
    while (!done && aabb.is_inside(x1)) {
      if (domain_check(x0) && domain_check(x1)) {
        sample1 = data_evaluator(x1);
        if ((sample0 <= isovalue && sample1 > isovalue) ||
            (sample0 >= isovalue && sample1 < isovalue)) {
          auto cur_x0      = x0;
          auto cur_x1      = x1;
          auto cur_sample0 = sample0;
          auto cur_sample1 = sample1;
          for (size_t i = 0; i < 100; ++i) {
            auto x_center      = (cur_x0 + cur_x1) / 2;
            auto sample_center = data_evaluator(x_center);
            if ((cur_sample0 <= isovalue && sample_center > isovalue) ||
                (cur_sample0 >= isovalue && sample_center < isovalue)) {
              cur_x1      = x_center;
              cur_sample1 = sample_center;
            } else {
              cur_x0      = x_center;
              cur_sample0 = sample_center;
            }
          }
          auto const t_iso =
              (isovalue - cur_sample0) / (cur_sample1 - cur_sample0);
          auto const iso_pos = r(t0 + t_iso * distance_on_ray);

          if constexpr (color_t::num_components() == 3) {
            accumulated_color = shader(iso_pos, r.direction());
            done              = true;
          } else if constexpr (color_t::num_components() == 4) {
            auto const rgba  = shader(iso_pos, r.direction());
            auto const rgb   = vec{rgba(0), rgba(1), rgba(2)};
            auto const alpha = rgba(3);
            accumulated_color += (1 - accumulated_alpha) * alpha * rgb;
            accumulated_alpha += (1 - accumulated_alpha) * alpha;
            if (accumulated_alpha >= 0.95) {
              done = true;
            }
          }
        }
      }
      t0      = t1;
      x0      = std::move(x1);
      sample0 = std::move(sample1);
      t1 += distance_on_ray;
      x1 = r(t1);
    }
    rendering(x, y) = accumulated_color * accumulated_alpha +
                      bg_color * (1 - accumulated_alpha);
  }
  return rendered_image;
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#ifdef __cpp_concepts
// template <arithmetic TReal, arithmetic Min, arithmetic Max,
//          arithmetic DistOnRay, arithmetic CameraReal, arithmetic AABBReal,
//          typename S, typename SReal, regular_invocable<SReal>    ColorScale,
//          regular_invocable<SReal> AlphaScale>
//#else
// template <
//    typename TReal, typename Min, typename Max, typename DistOnRay,
//    typename CameraReal, typename AABBReal, typename S, typename SReal,
//    typename ColorScale, typename AlphaScale,
//    enable_if<is_arithmetic<TReal, Min, Max, DistOnRay, CameraReal, AABBReal>,
//              is_invocable<ColorScale, SReal>,
//              is_invocable<AlphaScale, SReal>> = true>
//#endif
// auto direct_isosurface_rendering(
//    rendering::camera<CameraReal> const&          cam,
//    axis_aligned_bounding_box<AABBReal, 3> const& aabb,
//    scalarfield<S, SReal, 3> const& s, TReal const t, Min const min,
//    Max const max, DistOnRay const distance_on_ray, ColorScale&& color_scale,
//    AlphaScale&&                                   alpha_scale,
//    std::invoke_result_t<ColorScale, SReal> const& bg_color = {}) {
//  return direct_isosurface_rendering(
//      cam, aabb, [&](auto const& x) { return s(x, t); },
//      [&](auto const& x) { return s.in_domain(x, t); }, min, max,
//      distance_on_ray, std::forward<ColorScale>(color_scale),
//      std::forward<AlphaScale>(alpha_scale), bg_color);
//}
//// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
///-
//#ifdef __cpp_concepts
// template <arithmetic Min, arithmetic Max, arithmetic DistOnRay,
//          arithmetic CameraReal, typename Grid, typename ValueType,
//          bool HasNonConstReference, regular_invocable<double> ColorScale,
//          regular_invocable<double> AlphaScale>
//#else
// template <typename Min, typename Max, typename DistOnRay, typename
// CameraReal,
//          typename Grid, typename ValueType, bool HasNonConstReference,
//          typename ColorScale, typename AlphaScale,
//          enable_if<is_arithmetic<Min, Max, DistOnRay, CameraReal>,
//                    is_invocable<ColorScale, double>,
//                    is_invocable<AlphaScale, double>> = true>
//#endif
// auto direct_isosurface_rendering(
//    rendering::camera<CameraReal> const&                                  cam,
//    typed_grid_vertex_property_interface<Grid, ValueType, HasNonConstReference> const&
//    prop, Min const min, Max const max, DistOnRay const distance_on_ray,
//    ColorScale&& color_scale, AlphaScale&& alpha_scale,
//    std::invoke_result_t<ColorScale, ValueType> const& bg_color = {}) {
//  auto sampler = prop.template sampler<interpolation::cubic>();
//  return direct_isosurface_rendering(
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
