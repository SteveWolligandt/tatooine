/// \file direct_isosurface.h
/// This file specifies functions for direct renderings of iso surfaces.
#ifndef TATOOINE_RENDERING_DIRECT_ISOSURFACE_H
#define TATOOINE_RENDERING_DIRECT_ISOSURFACE_H
//==============================================================================
#include <tatooine/demangling.h>
#include <tatooine/field.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/rendering/camera.h>
//==============================================================================
namespace tatooine::rendering {
//==============================================================================
/// This is an implementation of \cite 10.5555/288216.288266. See also \ref
/// isosurf_parker.
///
/// \param cam Camera model for casting rays
/// \param linear_field Piece-wise trilinear field
/// \param isovalue Iso Value of the extracted iso surface
/// \param shader Shader for setting color at pixel. The shader takes position,
/// color (vec3 or vec4). \return Returns a 2D rectilinear grid with a
/// grid_vertex_property named "rendered_isosurface"
template <camera Camera, arithmetic IsoReal, typename Dim0, typename Dim1,
          typename Dim2, typename Field, typename Shader>
auto direct_isosurface(Camera const&                             cam,
                       rectilinear_grid<Dim0, Dim1, Dim2> const& g,
                       Field&& field, std::vector<IsoReal> const& isovalues,
                       Shader&& shader)
    //
    requires invocable<
        Shader, vec<typename rectilinear_grid<Dim0, Dim1, Dim2>::real_type, 3>,
        IsoReal, vec<typename rectilinear_grid<Dim0, Dim1, Dim2>::real_type, 3>,
        vec<typename Camera::real_type, 3>, vec<std::size_t, 2>>
//
{
  using grid_real_type = typename rectilinear_grid<Dim0, Dim1, Dim2>::real_type;
  using pos_type       = vec<grid_real_type, 3>;
  constexpr auto use_indices =
      std::is_invocable_v<Field, std::size_t, std::size_t, std::size_t>;
  // using value_type =
  //    std::conditional_t<use_indices,
  //                       std::invoke_result_t<Field, std::size_t, std::size_t,
  //                       std::size_t>, std::invoke_result_t<Field, pos_type>>;
  // static_assert(is_floating_point<value_type>);
  using cam_real_type = typename Camera::real_type;
  using viewdir_type  = vec<cam_real_type, 3>;
  using color_type    = invoke_result<Shader, pos_type, IsoReal, pos_type,
                                   viewdir_type, vec<std::size_t, 2>>;
  using rgb_type      = vec<typename color_type::value_type, 3>;
  using alpha_type    = typename color_type::value_type;
  using ray_type      = ray<cam_real_type, 3>;
  using pixel_pos     = Vec2<std::size_t>;
  static_assert(static_vec<color_type>,
                "Shader must return a vector with 3 or 4 components.");
  static_assert(
      color_type::num_components() == 3 || color_type::num_components() == 4,
      "Shader must return a vector with 3 or 4 components.");
  auto const& dim0 = g.template dimension<0>();
  auto const& dim1 = g.template dimension<1>();
  auto const& dim2 = g.template dimension<2>();
  auto const  aabb = g.bounding_box();
  auto        rendered_image =
      rectilinear_grid<linspace<cam_real_type>, linspace<cam_real_type>>{
          linspace<cam_real_type>{0.0, cam.plane_width() - 1,
                                  cam.plane_width()},
          linspace<cam_real_type>{0.0, cam.plane_height() - 1,
                                  cam.plane_height()}};
  auto& rendering =
      rendered_image.template vertex_property<rgb_type>("rendered_isosurface");
  //
  auto       rays     = std::vector<std::tuple<ray_type, double, pixel_pos>>{};
  auto       mutex    = std::mutex{};
  auto const bg_color = rgb_type{1, 1, 1};
  for_loop(
      [&](auto const... is) {
        rendering(is...) = bg_color;
        auto r           = cam.ray(is...);
        r.normalize();
        if (auto const i = aabb.check_intersection(r); i) {
          auto lock = std::lock_guard {mutex};
          rays.push_back(std::tuple{r, i->t, Vec2<std::size_t>{is...}});
        }
      },
      execution_policy::parallel, cam.plane_width(), cam.plane_height());
  for_loop(
      [&](auto const i) {
        auto cell_data         = std::make_unique<double[]>(8);
        auto indexing          = static_multidim_size<x_fastest, 2, 2, 2>{};
        auto accumulated_color = rgb_type::zeros();
        if constexpr (color_type::num_components() == 3) {
          accumulated_color = bg_color;
        }
        auto accumulated_alpha          = alpha_type{};
        auto const& [r, t, pixel_coord] = rays[i];

        auto entry_point = r(t);
        for (std::size_t i = 0; i < 3; ++i) {
          if (entry_point(i) < aabb.min(i)) {
            entry_point(i) = aabb.min(i);
          }
          if (entry_point(i) > aabb.max(i)) {
            entry_point(i) = aabb.max(i);
          }
        }

        auto cell_pos        = pos_type{};
        auto done            = false;
        auto update_cell_pos = [&](auto const& r) {
          for (std::size_t dim = 0; dim < 3; ++dim) {
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
          auto plane_indices_to_check = make_array<std::size_t, 3>();
          for (std::size_t dim = 0; dim < 3; ++dim) {
            if (cell_pos[dim] - std::floor(cell_pos[dim]) == 0) {
              if (r.direction(dim) > 0) {
                plane_indices_to_check[dim] =
                    static_cast<std::size_t>(cell_pos[dim]) + 1;
              } else {
                plane_indices_to_check[dim] =
                    static_cast<std::size_t>(cell_pos[dim]) - 1;
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
                 plane_indices_to_check[0] != std::size_t(0) - 1);
          assert(plane_indices_to_check[1] < dim1.size() &&
                 plane_indices_to_check[1] != std::size_t(0) - 1);
          assert(plane_indices_to_check[2] < dim2.size() &&
                 plane_indices_to_check[2] != std::size_t(0) - 1);
          auto const t0 =
              r.direction(0) == 0
                  ? std::numeric_limits<cam_real_type>::max()
                  : (dim0[plane_indices_to_check[0]] - r.origin(0)) /
                        r.direction(0);
          auto const t1 =
              r.direction(1) == 0
                  ? std::numeric_limits<cam_real_type>::max()
                  : (dim1[plane_indices_to_check[1]] - r.origin(1)) /
                        r.direction(1);
          auto const t2 =
              r.direction(2) == 0
                  ? std::numeric_limits<cam_real_type>::max()
                  : (dim2[plane_indices_to_check[2]] - r.origin(2)) /
                        r.direction(2);

          std::array<std::size_t, 3> i0{0, 0, 0};
          std::array<std::size_t, 3> i1{0, 0, 0};
          for (std::size_t dim = 0; dim < 3; ++dim) {
            if (cell_pos[dim] - std::floor(cell_pos[dim]) == 0) {
              if (r.direction(dim) > 0) {
                i0[dim] = static_cast<std::size_t>(cell_pos[dim]);
                i1[dim] = static_cast<std::size_t>(cell_pos[dim]) + 1;
              } else if (r.direction(dim) < 0) {
                i0[dim] = static_cast<std::size_t>(cell_pos[dim]) - 1;
                i1[dim] = static_cast<std::size_t>(cell_pos[dim]);
              }
            } else {
              i0[dim] = static_cast<std::size_t>(std::floor(cell_pos[dim]));
              i1[dim] = static_cast<std::size_t>(std::ceil(cell_pos[dim]));
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
            cell_data[indexing(0, 0, 0)] =
                field(g.vertex_at(i0[0], i0[1], i0[2]));
            cell_data[indexing(1, 0, 0)] =
                field(g.vertex_at(i1[0], i0[1], i0[2]));
            cell_data[indexing(0, 1, 0)] =
                field(g.vertex_at(i0[0], i1[1], i0[2]));
            cell_data[indexing(1, 1, 0)] =
                field(g.vertex_at(i1[0], i1[1], i0[2]));
            cell_data[indexing(0, 0, 1)] =
                field(g.vertex_at(i0[0], i0[1], i1[2]));
            cell_data[indexing(1, 0, 1)] =
                field(g.vertex_at(i1[0], i0[1], i1[2]));
            cell_data[indexing(0, 1, 1)] =
                field(g.vertex_at(i0[0], i1[1], i1[2]));
            cell_data[indexing(1, 1, 1)] =
                field(g.vertex_at(i1[0], i1[1], i1[2]));
          }

          auto const  x0              = g.vertex_at(i0[0], i0[1], i0[2]);
          auto const  x1              = g.vertex_at(i1[0], i1[1], i1[2]);
          auto const& xa              = r.origin();
          auto const& xb              = r.direction();
          auto const  cell_extent     = x1 - x0;
          auto const  inv_cell_extent = pos_type{
              1 / cell_extent(0), 1 / cell_extent(1), 1 / cell_extent(2)};
          // create rays in different spaces
          auto const a0 = (x1 - xa) * inv_cell_extent;
          auto const b0 = xb * inv_cell_extent;
          auto const a1 = (xa - x0) * inv_cell_extent;
          auto const b1 = -xb * inv_cell_extent;

          std::vector<std::tuple<grid_real_type, IsoReal, pos_type>>
              found_surfaces;
          for (auto const isovalue : isovalues) {
            // check if isosurface is present in current cell
            if (!((cell_data[indexing(0, 0, 0)] > isovalue &&
                   cell_data[indexing(0, 0, 1)] > isovalue &&
                   cell_data[indexing(0, 1, 0)] > isovalue &&
                   cell_data[indexing(0, 1, 1)] > isovalue &&
                   cell_data[indexing(1, 0, 0)] > isovalue &&
                   cell_data[indexing(1, 0, 1)] > isovalue &&
                   cell_data[indexing(1, 1, 0)] > isovalue &&
                   cell_data[indexing(1, 1, 1)] > isovalue) ||
                  (cell_data[indexing(0, 0, 0)] < isovalue &&
                   cell_data[indexing(0, 0, 1)] < isovalue &&
                   cell_data[indexing(0, 1, 0)] < isovalue &&
                   cell_data[indexing(0, 1, 1)] < isovalue &&
                   cell_data[indexing(1, 0, 0)] < isovalue &&
                   cell_data[indexing(1, 0, 1)] < isovalue &&
                   cell_data[indexing(1, 1, 0)] < isovalue &&
                   cell_data[indexing(1, 1, 1)] < isovalue))) {
              // construct coefficients of cubic polynomial A + B*t + C*t*t +
              // D*t*t*t
              auto const A =
                  a0(0) * a0(1) * a0(2) * cell_data[indexing(0, 0, 0)] +
                  a0(0) * a0(1) * a1(2) * cell_data[indexing(0, 0, 1)] +
                  a0(0) * a1(1) * a0(2) * cell_data[indexing(0, 1, 0)] +
                  a0(0) * a1(1) * a1(2) * cell_data[indexing(0, 1, 1)] +
                  a1(0) * a0(1) * a0(2) * cell_data[indexing(1, 0, 0)] +
                  a1(0) * a0(1) * a1(2) * cell_data[indexing(1, 0, 1)] +
                  a1(0) * a1(1) * a0(2) * cell_data[indexing(1, 1, 0)] +
                  a1(0) * a1(1) * a1(2) * cell_data[indexing(1, 1, 1)] -
                  isovalue;
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
              auto const D =
                  b0(0) * b0(1) * b0(2) * cell_data[indexing(0, 0, 0)] +
                  b0(0) * b0(1) * b1(2) * cell_data[indexing(0, 0, 1)] +
                  b0(0) * b1(1) * b0(2) * cell_data[indexing(0, 1, 0)] +
                  b0(0) * b1(1) * b1(2) * cell_data[indexing(0, 1, 1)] +
                  b1(0) * b0(1) * b0(2) * cell_data[indexing(1, 0, 0)] +
                  b1(0) * b0(1) * b1(2) * cell_data[indexing(1, 0, 1)] +
                  b1(0) * b1(1) * b0(2) * cell_data[indexing(1, 1, 0)] +
                  b1(0) * b1(1) * b1(2) * cell_data[indexing(1, 1, 1)];

              auto const s = solve(polynomial{A, B, C, D});

              if (!s.empty()) {
                for (auto const t : s) {
                  constexpr auto eps = 1e-10;
                  if (auto x = a0 + t * b0;                //
                      - eps <= x(0) && x(0) <= 1 + eps &&  //
                      -eps <= x(1) && x(1) <= 1 + eps &&   //
                      -eps <= x(2) && x(2) <= 1 + eps) {
                    found_surfaces.emplace_back(t, isovalue, x);
                  }
                }
              }
            }
          }
          std::sort(begin(found_surfaces), end(found_surfaces),
                    [](auto const& s0, auto const& s1) {
                      auto const& [t0, iso0, uvw0] = s0;
                      auto const& [t1, iso1, uvw1] = s1;
                      return t0 < t1;
                    });
          for (auto const& [t, isovalue, uvw1] : found_surfaces) {
            auto const uvw0  = pos_type{1 - uvw1(0), 1 - uvw1(1), 1 - uvw1(2)};
            auto const x_iso = uvw0 * cell_extent + x0;
            assert(uvw0(0) >= 0 && uvw0(0) <= 1);
            assert(uvw0(1) >= 0 && uvw0(1) <= 1);
            assert(uvw0(2) >= 0 && uvw0(2) <= 1);
            assert(uvw1(0) >= 0 && uvw1(0) <= 1);
            assert(uvw1(1) >= 0 && uvw1(1) <= 1);
            assert(uvw1(2) >= 0 && uvw1(2) <= 1);
            auto const k =
                cell_data[indexing(1, 1, 1)] - cell_data[indexing(0, 1, 1)] -
                cell_data[indexing(1, 0, 1)] + cell_data[indexing(0, 0, 1)] -
                cell_data[indexing(1, 1, 0)] + cell_data[indexing(0, 1, 0)] +
                cell_data[indexing(1, 0, 0)] - cell_data[indexing(0, 0, 0)];
            auto const gradient = vec{
                (k * uvw0(1) + cell_data[indexing(1, 0, 1)] -
                 cell_data[indexing(0, 0, 1)] - cell_data[indexing(1, 0, 0)] +
                 cell_data[indexing(0, 0, 0)]) *
                        uvw0(2) +
                    (cell_data[indexing(1, 1, 0)] -
                     cell_data[indexing(0, 1, 0)] -
                     cell_data[indexing(1, 0, 0)] +
                     cell_data[indexing(0, 0, 0)]) *
                        uvw0(1) +
                    cell_data[indexing(1, 0, 0)] - cell_data[indexing(0, 0, 0)],
                (k * uvw0(0) + cell_data[indexing(0, 1, 1)] -
                 cell_data[indexing(0, 0, 1)] - cell_data[indexing(0, 1, 0)] +
                 cell_data[indexing(0, 0, 0)]) *
                        uvw0(2) +
                    (cell_data[indexing(1, 1, 0)] -
                     cell_data[indexing(0, 1, 0)] -
                     cell_data[indexing(1, 0, 0)] +
                     cell_data[indexing(0, 0, 0)]) *
                        uvw0(0) +
                    cell_data[indexing(0, 1, 0)] - cell_data[indexing(0, 0, 0)],
                (k * uvw0(0) + cell_data[indexing(0, 1, 1)] -
                 cell_data[indexing(0, 0, 1)] - cell_data[indexing(0, 1, 0)] +
                 cell_data[indexing(0, 0, 0)]) *
                        uvw0(1) +
                    (cell_data[indexing(1, 0, 1)] -
                     cell_data[indexing(0, 0, 1)] -
                     cell_data[indexing(1, 0, 0)] +
                     cell_data[indexing(0, 0, 0)]) *
                        uvw0(0) +
                    cell_data[indexing(0, 0, 1)] -
                    cell_data[indexing(0, 0, 0)]};
            if constexpr (color_type::num_components() == 3) {
              accumulated_color =
                  shader(x_iso, isovalue, gradient, r.direction(), pixel_coord);
              done = true;
            } else if constexpr (color_type::num_components() == 4) {
              auto const rgba =
                  shader(x_iso, isovalue, gradient, r.direction(), pixel_coord);
              auto const rgb   = vec{rgba(0), rgba(1), rgba(2)};
              auto const alpha = rgba(3);
              accumulated_color += (1 - accumulated_alpha) * alpha * rgb;
              accumulated_alpha += (1 - accumulated_alpha) * alpha;
              if (accumulated_alpha >= 0.95) {
                done = true;
              }
            }
          }

          if (!done) {
            entry_point = r(tatooine::min(t0, t1, t2));
            update_cell_pos(r);
          }
        }
        if constexpr (color_type::num_components() == 3) {
          rendering(pixel_coord.x(), pixel_coord.y()) = accumulated_color;
        } else if constexpr (color_type::num_components() == 4) {
          rendering(pixel_coord.x(), pixel_coord.y()) =
              accumulated_color * accumulated_alpha +
              bg_color * (1 - accumulated_alpha);
        }
      },
      execution_policy::parallel, rays.size());
  return rendered_image;
}
//==============================================================================
template <typename IsoReal, typename Dim0, typename Dim1, typename Dim2,
          typename Field, typename FieldReal, typename Shader>
auto direct_isosurface(camera auto const&                        cam,
                       rectilinear_grid<Dim0, Dim1, Dim2> const& g,
                       scalarfield<Field, FieldReal, 3> const&        field,
                       IsoReal const isovalue, Shader&& shader) {
  return direct_isosurface(cam, g, field, std::vector{isovalue},
                           std::forward<Shader>(shader));
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
template <camera     Camera, typename GridVertexProperty, typename Shader,
          arithmetic Iso>
auto direct_isosurface(
    Camera const& cam,
    tatooine::detail::rectilinear_grid::vertex_property_sampler<
        GridVertexProperty, interpolation::linear, interpolation::linear,
        interpolation::linear> const& linear_field,
    Iso const isovalue, Shader&& shader)
    //
    requires invocable<
        Shader, vec<typename GridVertexProperty::grid_type::real_type, 3>,
        Iso, vec<typename GridVertexProperty::grid_type::real_type, 3>,
        vec<typename Camera::real_type, 3>, vec<std::size_t, 2>>
//
{
  return direct_isosurface(
      cam, linear_field.grid(),
      [&](std::size_t const ix, std::size_t const iy, std::size_t const iz)
          -> auto const& { return linear_field.data_at(ix, iy, iz); },
      std::vector{isovalue}, std::forward<Shader>(shader));
}
//------------------------------------------------------------------------------
template <typename DistOnRay, typename AABBReal, typename DataEvaluator,
          typename Isovalue, typename DomainCheck, typename Shader>
auto direct_isosurface(camera auto const&                            cam,
                       axis_aligned_bounding_box<AABBReal, 3> const& aabb,
                       DataEvaluator&& data_evaluator,
                       DomainCheck&& domain_check, Isovalue isovalue,
                       DistOnRay const distance_on_ray, Shader&& shader) {
  using cam_real_type = typename std::decay_t<decltype(cam)>::real_type;
  using pos_type      = vec<cam_real_type, 3>;
  using viewdir_type  = vec<cam_real_type, 3>;
  static_assert(
      std::is_invocable_v<Shader, pos_type, viewdir_type, vec<std::size_t, 2>>,
      "Shader must be invocable with position and view direction.");
  using value_type = std::invoke_result_t<DataEvaluator, pos_type>;
  using color_type = std::invoke_result_t<Shader, pos_type, viewdir_type>;
  using rgb_type   = vec<typename color_type::value_type, 3>;
  using alpha_type = typename color_type::value_type;
  using ray_type   = ray<cam_real_type, 3>;
  static_assert(is_floating_point<value_type>,
                "DataEvaluator must return scalar type.");
  static_assert(static_vec<color_type>,
                "ColorScale must return scalar type or tatooine::vec.");
  static_assert(
      color_type::num_components() == 3 || color_type::num_components() == 4,
      "ColorScale must return scalar type or tatooine::vec.");
  auto rendered_image =
      rectilinear_grid<linspace<cam_real_type>, linspace<cam_real_type>>{
          linspace<cam_real_type>{0.0, cam.plane_width() - 1,
                                  cam.plane_width()},
          linspace<cam_real_type>{0.0, cam.plane_height() - 1,
                                  cam.plane_height()}};
  auto& rendering =
      rendered_image.template vertex_property<rgb_type>("rendered_isosurface");

  auto rays = std::vector<
      std::tuple<ray_type, AABBReal, std::size_t, std::size_t>>{};
  auto       mutex    = std::mutex{};
  auto const bg_color = rgb_type{1, 1, 1};
  for_loop(
      [&](auto const... is) {
        rendering(is...) = bg_color;
        auto r           = cam.ray(is...);
        r.normalize();
        if (auto const i = aabb.check_intersection(r); i) {
          std::lock_guard lock{mutex};
          rays.push_back(std::tuple{r, i->t, is...});
        }
      },
      execution_policy::parallel, cam.plane_width(), cam.plane_height());
  for_loop(
      [&](auto const i) {
        auto const [r, t, x, y]      = rays[i];
        auto accumulated_alpha = alpha_type{};
        auto accumulated_color       = rgb_type{};

        auto t0 = t;
        auto x0 = r(t0);
        for (std::size_t i = 0; i < 3; ++i) {
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
              for (std::size_t i = 0; i < 100; ++i) {
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

              if constexpr (color_type::num_components() == 3) {
                accumulated_color = shader(iso_pos, r.direction());
                done              = true;
              } else if constexpr (color_type::num_components() == 4) {
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
      },
      execution_policy::parallel, rays.size());
  return rendered_image;
}
//==============================================================================
}  // namespace tatooine::rendering
//==============================================================================
#endif
