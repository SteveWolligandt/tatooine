#ifndef TATOOINE_ISOLINES_H
#define TATOOINE_ISOLINES_H
//==============================================================================
#include <cassert>
#include <string>
#include <vector>
#ifndef NDEBUG
#include <mutex>
#endif

#include <tatooine/field.h>
#include <tatooine/for_loop.h>
#include <tatooine/multidim_array.h>
#include <tatooine/tensor.h>
#include <tatooine/utility.h>
#include <tatooine/edgeset.h>
#include <tatooine/rectilinear_grid.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief      Indexing and lookup map from
/// http://paulbourke.net/geometry/polygonise/
template <typename XDomain, typename YDomain>
auto isolines(
    invocable<
        std::size_t, std::size_t,
        Vec2<typename rectilinear_grid<XDomain, YDomain>::real_type>> auto&&
                                              get_scalars,
    rectilinear_grid<XDomain, YDomain> const& g,
    arithmetic auto const                     isolevel) {
  using real_type    = typename rectilinear_grid<XDomain, YDomain>::real_type;
  using edgeset_type = Edgeset2<real_type>;
  auto isolines      = edgeset_type{};

#ifdef NDEBUG
  std::mutex mutex;
#endif
  auto process_cell = [&](auto ix, auto iy) {
    auto       iso_positions = std::vector<typename edgeset_type::vertex_handle>{};
    auto       p =
        std::array{g.vertex_at(ix, iy), g.vertex_at(ix + 1, iy), g.vertex_at(ix + 1, iy + 1), g.vertex_at(ix, iy + 1)};

    auto s = std::array{
        get_scalars(ix, iy, p[0]), get_scalars(ix + 1, iy, p[1]),
        get_scalars(ix + 1, iy + 1, p[2]), get_scalars(ix, iy + 1, p[3])};
    if (s[0] > isolevel && s[1] > isolevel && s[2] > isolevel &&
        s[3] > isolevel) {
      return;
    }
    if (s[0] < isolevel && s[1] < isolevel && s[2] < isolevel &&
        s[3] < isolevel) {
      return;
    }

    auto check_edge = [&](std::size_t const i, std::size_t const j) {
      if ((s[i] > isolevel && s[j] < isolevel) ||
          (s[i] < isolevel && s[j] > isolevel)) {
        auto const t = (isolevel - s[i]) / (-s[i] + s[j]);
        iso_positions.push_back(isolines.insert_vertex(p[i] * (1 - t) + p[j] * t));
      }
    };
    check_edge(0, 1);
    check_edge(1, 2);
    check_edge(2, 3);
    check_edge(3, 0);
    for (std::size_t i = 0; i < 4; ++i) {
      if (s[i] == isolevel) {
        iso_positions.push_back(isolines.insert_vertex(p[i]));
      }
    }

    if (size(iso_positions) == 2) {
#ifdef NDEBUG
      {
        std::lock_guard lock{mutex};
#endif
        isolines.insert_edge(iso_positions[0], iso_positions[1]);
#ifdef NDEBUG
      }
#endif
    }
    if (size(iso_positions) == 4) {
      auto const scalar_center = (s[0] + s[1] + s[2] + s[3]) / 4;
#ifdef NDEBUG
      {
        std::lock_guard lock{mutex};
#endif
        if (scalar_center > isolevel && s[0] > isolevel) {
          isolines.insert_edge(iso_positions[0], iso_positions[1]);
          isolines.insert_edge(iso_positions[2], iso_positions[3]);
        } else {
          isolines.insert_edge(iso_positions[0], iso_positions[3]);
          isolines.insert_edge(iso_positions[2], iso_positions[1]);
        }
#ifdef NDEBUG
      }
#endif
    }
  };
  auto execution =
#ifdef NDEBUG
      execution_policy::sequential;
#else
      execution_policy::sequential;
#endif
  for_loop(process_cell, execution, g.size(0) - 1, g.size(1) - 1);
  return isolines;
}
//------------------------------------------------------------------------------
template <typename Grid, arithmetic T, bool HasNonConstReference>
auto isolines(detail::rectilinear_grid::typed_vertex_property_interface<
                  Grid, T, HasNonConstReference> const& data,
              arithmetic auto const                     isolevel) {
  return isolines(
      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
        return data(ix, iy);
      },
      data.grid(), isolevel);
}
//------------------------------------------------------------------------------
template <arithmetic Real, typename Indexing, arithmetic BBReal>
auto isolines(dynamic_multidim_array<Real, Indexing> const& data,
              axis_aligned_bounding_box<BBReal, 2> const&   bb,
              arithmetic auto const                         isolevel) {
  assert(data.num_dimensions() == 2);
  return isolines(
      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
        return data(ix, iy);
      },
      rectilinear_grid{linspace{bb.min(0), bb.max(0), data.size(0)},
                       linspace{bb.min(1), bb.max(1), data.size(1)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <arithmetic Real, arithmetic  BBReal, typename Indexing,
          typename MemLoc, std::size_t XRes, std::size_t YRes>
auto isolines(
    static_multidim_array<Real, Indexing, MemLoc, XRes, YRes> const& data,
    axis_aligned_bounding_box<BBReal, 2> const&                      bb,
    arithmetic auto const                                            isolevel) {
  return isolines(
      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
        return data(ix, iy);
      },
      rectilinear_grid{linspace{bb.min(0), bb.max(0), data.size(0)},
                       linspace{bb.min(1), bb.max(1), data.size(1)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <typename Field, typename FieldReal,
          floating_point_range XDomain,
          floating_point_range YDomain,
          arithmetic                          TReal = FieldReal>
auto isolines(scalarfield<Field, FieldReal, 2> const&         sf,
              rectilinear_grid<XDomain, YDomain> const& g,
              arithmetic auto const isolevel, TReal const t = 0) {
  auto eval = [&](auto const /*ix*/, auto const /*iy*/, auto const& pos) {
    return sf(pos, t);
  };
  return isolines(eval, g, isolevel);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
