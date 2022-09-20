#ifndef TATOOINE_ISOSURFACE_H
#define TATOOINE_ISOSURFACE_H
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
#include <tatooine/unstructured_triangular_grid.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief      Indexing and lookup map from
/// http://paulbourke.net/geometry/polygonise/
template <
    typename XDomain, typename YDomain, typename ZDomain, arithmetic Isolevel,
    invocable<
        std::size_t, std::size_t, std::size_t,
        vec<typename rectilinear_grid<XDomain, YDomain, ZDomain>::real_type,
            3> >
        GetScalars>
auto isosurface(GetScalars&&                                       get_scalars,
                rectilinear_grid<XDomain, YDomain, ZDomain> const& g,
                Isolevel const                                     isolevel) {
  using real_type =
      typename rectilinear_grid<XDomain, YDomain, ZDomain>::real_type;
  using pos_type = vec<real_type, 3>;
  unstructured_triangular_grid<real_type, 3> iso_volume;

#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  std::mutex mutex;
#endif
  auto process_cube = [&](auto ix, auto iy, auto iz) {
    auto       vertlist = make_array<pos_type, 12>();
    std::array p{g.vertex_at(ix, iy, iz + 1),     g.vertex_at(ix + 1, iy, iz + 1),
                 g.vertex_at(ix + 1, iy, iz),     g.vertex_at(ix, iy, iz),
                 g.vertex_at(ix, iy + 1, iz + 1), g.vertex_at(ix + 1, iy + 1, iz + 1),
                 g.vertex_at(ix + 1, iy + 1, iz), g.vertex_at(ix, iy + 1, iz)};

    decltype(auto) s0 = get_scalars(ix, iy, iz + 1, p[0]);
    decltype(auto) s1 = get_scalars(ix + 1, iy, iz + 1, p[1]);
    decltype(auto) s2 = get_scalars(ix + 1, iy, iz, p[2]);
    decltype(auto) s3 = get_scalars(ix, iy, iz, p[3]);
    decltype(auto) s4 = get_scalars(ix, iy + 1, iz + 1, p[4]);
    decltype(auto) s5 = get_scalars(ix + 1, iy + 1, iz + 1, p[5]);
    decltype(auto) s6 = get_scalars(ix + 1, iy + 1, iz, p[6]);
    decltype(auto) s7 = get_scalars(ix, iy + 1, iz, p[7]);

    unsigned int cube_index = 0;
    if (s0 < isolevel) {
      cube_index |= 1;
    }
    if (s1 < isolevel) {
      cube_index |= 2;
    }
    if (s2 < isolevel) {
      cube_index |= 4;
    }
    if (s3 < isolevel) {
      cube_index |= 8;
    }
    if (s4 < isolevel) {
      cube_index |= 16;
    }
    if (s5 < isolevel) {
      cube_index |= 32;
    }
    if (s6 < isolevel) {
      cube_index |= 64;
    }
    if (s7 < isolevel) {
      cube_index |= 128;
    }

    // Cube is entirely in/out of the surface
    if (marchingcubes_lookup::edge_table[cube_index] == 0) {
      return;
    }

    // Find the vertices where the surface intersects the cube
    if (marchingcubes_lookup::edge_table[cube_index] & 1) {
      real_type const s = (isolevel - s0) / (s1 - s0);
      vertlist[0]       = p[0] * (1 - s) + p[1] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 2) {
      real_type const s = (isolevel - s1) / (s2 - s1);
      vertlist[1]       = p[1] * (1 - s) + p[2] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 4) {
      real_type const s = (isolevel - s2) / (s3 - s2);
      vertlist[2]       = p[2] * (1 - s) + p[3] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 8) {
      real_type const s = (isolevel - s3) / (s0 - s3);
      vertlist[3]       = p[3] * (1 - s) + p[0] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 16) {
      real_type const s = (isolevel - s4) / (s5 - s4);
      vertlist[4]       = p[4] * (1 - s) + p[5] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 32) {
      real_type const s = (isolevel - s5) / (s6 - s5);
      vertlist[5]       = p[5] * (1 - s) + p[6] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 64) {
      real_type const s = (isolevel - s6) / (s7 - s6);
      vertlist[6]       = p[6] * (1 - s) + p[7] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 128) {
      real_type const s = (isolevel - s7) / (s4 - s7);
      vertlist[7]       = p[7] * (1 - s) + p[4] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 256) {
      real_type const s = (isolevel - s0) / (s4 - s0);
      vertlist[8]       = p[0] * (1 - s) + p[4] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 512) {
      real_type const s = (isolevel - s1) / (s5 - s1);
      vertlist[9]       = p[1] * (1 - s) + p[5] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 1024) {
      real_type const s = (isolevel - s2) / (s6 - s2);
      vertlist[10]      = p[2] * (1 - s) + p[6] * s;
    }
    if (marchingcubes_lookup::edge_table[cube_index] & 2048) {
      real_type const s = (isolevel - s3) / (s7 - s3);
      vertlist[11]      = p[3] * (1 - s) + p[7] * s;
    }

#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
    {
      std::lock_guard lock{mutex};
#endif
      // create the triangle
      for (std::size_t i = 0;
           marchingcubes_lookup::tri_table[cube_index][i] != -1; i += 3) {
        iso_volume.insert_simplex(
            iso_volume.insert_vertex(
                vertlist[marchingcubes_lookup::tri_table[cube_index][i]]),
            iso_volume.insert_vertex(
                vertlist[marchingcubes_lookup::tri_table[cube_index][i + 2]]),
            iso_volume.insert_vertex(
                vertlist[marchingcubes_lookup::tri_table[cube_index][i + 1]]));
      }
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
    }
#endif
  };
#if defined(NDEBUG) && defined(TATOOINE_OPENMP_AVAILABLE)
  for_loop(process_cube, execution_policy::parallel, g.size(0) - 1,
           g.size(1) - 1, g.size(2) - 1);
#else
  for_loop(process_cube, execution_policy::sequential, g.size(0) - 1,
           g.size(1) - 1, g.size(2) - 1);
#endif
  return iso_volume;
}
//------------------------------------------------------------------------------
template <arithmetic Real, typename Indexing, arithmetic BBReal,
          arithmetic Isolevel>
auto isosurface(dynamic_multidim_array<Real, Indexing> const& data,
                axis_aligned_bounding_box<BBReal, 3> const&   bb,
                Isolevel const                                isolevel) {
  assert(data.num_dimensions() == 3);
  return isosurface(
      [&](auto ix, auto iy, auto iz, auto const & /*ps*/) -> auto const& {
        return data(ix, iy, iz);
      },
      rectilinear_grid{linspace{bb.min(0), bb.max(0), data.size(0)},
                       linspace{bb.min(1), bb.max(1), data.size(1)},
                       linspace{bb.min(2), bb.max(2), data.size(2)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <arithmetic Real, typename Indexing, typename MemLoc, std::size_t XRes,
          std::size_t YRes, std::size_t ZRes, arithmetic BBReal,
          arithmetic Isolevel>
auto isosurface(
    static_multidim_array<Real, Indexing, MemLoc, XRes, YRes, ZRes> const& data,
    axis_aligned_bounding_box<BBReal, 3> const& bb, Isolevel const isolevel) {
  return isosurface(
      [&](auto ix, auto iy, auto iz, auto const & /*ps*/) -> auto const& {
        return data(ix, iy, iz);
      },
      rectilinear_grid{linspace{bb.min(0), bb.max(0), data.size(0)},
                       linspace{bb.min(1), bb.max(1), data.size(1)},
                       linspace{bb.min(2), bb.max(2), data.size(2)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <typename Field, arithmetic FieldReal, typename XDomain,
          typename YDomain, typename ZDomain, arithmetic Isolevel,
          arithmetic TReal = int>
auto isosurface(scalarfield<Field, FieldReal, 3> const&            sf,
                rectilinear_grid<XDomain, YDomain, ZDomain> const& g,
                Isolevel const isolevel, TReal const t = 0) {
  return isosurface(
      [&](integral auto const /*ix*/, integral auto const /*iy*/,
          integral auto const /*iz*/, auto const& pos) { return sf(pos, t); },
      g, isolevel);
}
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
template <typename Grid, arithmetic T, bool HasNonConstReference>
auto isosurface(detail::rectilinear_grid::typed_vertex_property_interface<
                  Grid, T, HasNonConstReference> const& data,
              arithmetic auto const                     isolevel) {
  return isosurface([&](integral auto const ix, integral auto const iy,
                        integral auto const iz,
                        auto const& /*pos*/) { return data(ix, iy, iz); },
                    data.grid(), isolevel);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
