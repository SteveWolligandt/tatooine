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
#include <tatooine/marchingcubeslookuptable.h>
#include <tatooine/multidim_array.h>
#include <tatooine/tensor.h>
#include <tatooine/triangular_mesh.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief      Indexing and lookup map from
/// http://paulbourke.net/geometry/polygonise/
template <indexable_space XDomain, indexable_space YDomain,
          invocable<size_t const, size_t const,
                    vec<typename grid<XDomain, YDomain>::real_t, 2> const&>
              GetScalars>
auto isolines(GetScalars&& get_scalars, grid<XDomain, YDomain> const& g,
              real_number auto const isolevel) {
  using real_t = typename grid<XDomain, YDomain>::real_t;
  using pos_t = vec<real_t, 2>;
  using edge_set_t = std::vector<line<real_t, 2>>;
  edge_set_t isolines;

#ifdef NDEBUG
  std::mutex mutex;
#endif
  auto process_cube = [&](auto ix, auto iy) {
    std::vector<pos_t> iso_positions;
    std::array p{g(ix,     iy),
                 g(ix + 1, iy),
                 g(ix + 1, iy + 1),
                 g(ix,     iy + 1)};

    std::array s{get_scalars(ix,     iy,     p[0]),
                 get_scalars(ix + 1, iy,     p[1]),
                 get_scalars(ix + 1, iy + 1, p[2]),
                 get_scalars(ix,     iy + 1, p[3])};
    if (s[0] > isolevel &&
        s[1] > isolevel &&
        s[2] > isolevel &&
        s[3] > isolevel) {
      return;
    }
    if (s[0] <= isolevel &&
        s[1] <= isolevel &&
        s[2] <= isolevel &&
        s[3] <= isolevel) {
      return;
    }

    auto check_edge = [&](size_t const i, size_t const j) {
      if ((s[i] > isolevel && s[j] < isolevel) ||
          (s[i] < isolevel && s[j] > isolevel)) {
        auto const t = (isolevel - s[i]) / (-s[i] + s[j]);
        iso_positions.push_back(p[i] * (1 - t) + p[j] * t);
      }
    };
    check_edge(0, 1);
    check_edge(1, 2);
    check_edge(2, 3);
    check_edge(3, 0);

    if (size(iso_positions) == 2) {
#ifdef NDEBUG
      {
        std::lock_guard lock{mutex};
#endif
        isolines.emplace_back(iso_positions[0], iso_positions[1]);
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
          isolines.emplace_back(iso_positions[0], iso_positions[1]);
          isolines.emplace_back(iso_positions[2], iso_positions[3]);
        } else {
          isolines.emplace_back(iso_positions[0], iso_positions[3]);
          isolines.emplace_back(iso_positions[2], iso_positions[1]);
        }
#ifdef NDEBUG
      }
#endif
    }
  };
#ifdef NDEBUG
  parallel_for_loop(process_cube, g.size(0) - 1, g.size(1) - 1);
#else
  for_loop(process_cube, g.size(0) - 1, g.size(1) - 1);
#endif
  return isolines;
}
//------------------------------------------------------------------------------
template <typename Grid, real_number T>
auto isolines(typed_multidim_property<Grid, T> const& data,
              real_number auto const                  isolevel) {
  return isolines(
      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
        return data(ix, iy);
      },
      data.grid(), isolevel);
}
//------------------------------------------------------------------------------
template <real_number Real, typename Indexing, real_number BBReal>
auto isolines(dynamic_multidim_array<Real, Indexing> const& data,
              axis_aligned_bounding_box<BBReal, 2> const&   bb,
              real_number auto const                        isolevel) {
  assert(data.num_dimensions() == 2);
  return isolines(
      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
        return data(ix, iy);
      },
      grid{linspace{bb.min(0), bb.max(0), data.size(0)},
           linspace{bb.min(1), bb.max(1), data.size(1)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <real_number Real, real_number BBReal, typename Indexing, typename MemLoc, size_t XRes,
          size_t YRes>
auto isolines(
    static_multidim_array<Real, Indexing, MemLoc, XRes, YRes> const& data,
    axis_aligned_bounding_box<BBReal, 2> const& bb, real_number auto const isolevel) {
  return isolines(
      [&](auto ix, auto iy, auto const& /*ps*/) -> auto const& {
        return data(ix, iy);
      },
      grid{linspace{bb.min(0), bb.max(0), data.size(0)},
           linspace{bb.min(1), bb.max(1), data.size(1)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <typename Field, typename FieldReal, indexable_space XDomain,
          indexable_space YDomain, real_number TReal = FieldReal>
auto isolines(field<Field, FieldReal, 2> const& sf,
              grid<XDomain, YDomain> const& g, real_number auto const isolevel,
              TReal const t = 0) {
  auto eval = [&](auto const /*ix*/, auto const /*iy*/, auto const& pos) {
    return sf(pos, t);
  };
  return isolines(eval, g, isolevel);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
