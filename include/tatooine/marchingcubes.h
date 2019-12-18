#ifndef TATOOINE_MARCHING_CUBES_H
#define TATOOINE_MARCHING_CUBES_H


#include <cassert>
#include <string>
#include <vector>
#ifndef NDEBUG
#include <mutex>
#endif

#include "field.h"
#include "multidim_array.h"
#include "marchingcubeslookuptable.h"
#include "simple_tri_mesh.h"
#include "tensor.h"
#include "nested_for_loop.h"
#include "utility.h"

//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief      Indexing and lookup map from
/// http://paulbourke.net/geometry/polygonise/
template <typename Real, typename GetScalars, typename GridReal, typename IsolevelReal>
auto marchingcubes(GetScalars&& get_scalars, const grid<GridReal, 3>& g,
                   IsolevelReal isolevel) {
  using pos_t = vec<Real, 3>;
  simple_tri_mesh<Real, 3> iso_volume;

#ifdef NDEBUG
  std::mutex mutex;
#endif
  auto process_cube = [&](auto ix, auto iy, auto iz) {
    auto  vertlist = make_array<pos_t, 12>(pos_t::zeros());
    std::array p{g(ix, iy, iz + 1),     g(ix + 1, iy, iz + 1),
                 g(ix + 1, iy, iz),     g(ix, iy, iz),
                 g(ix, iy + 1, iz + 1), g(ix + 1, iy + 1, iz + 1),
                 g(ix + 1, iy + 1, iz), g(ix, iy + 1, iz)};

    const decltype(auto) s0 = get_scalars(ix, iy, iz + 1, p[0]);
    const decltype(auto) s1 = get_scalars(ix + 1, iy, iz + 1, p[1]);
    const decltype(auto) s2 = get_scalars(ix + 1, iy, iz, p[2]);
    const decltype(auto) s3 = get_scalars(ix, iy, iz, p[3]);
    const decltype(auto) s4 = get_scalars(ix, iy + 1, iz + 1, p[4]);
    const decltype(auto) s5 = get_scalars(ix + 1, iy + 1, iz + 1, p[5]);
    const decltype(auto) s6 = get_scalars(ix + 1, iy + 1, iz, p[6]);
    const decltype(auto) s7 = get_scalars(ix, iy + 1, iz, p[7]);

    unsigned int cube_index = 0;
    if (s0 < isolevel) { cube_index |= 1; }
    if (s1 < isolevel) { cube_index |= 2; }
    if (s2 < isolevel) { cube_index |= 4; }
    if (s3 < isolevel) { cube_index |= 8; }
    if (s4 < isolevel) { cube_index |= 16; }
    if (s5 < isolevel) { cube_index |= 32; }
    if (s6 < isolevel) { cube_index |= 64; }
    if (s7 < isolevel) { cube_index |= 128; }

    // Cube is entirely in/out of the surface
    if (marchingcubes_lookup::edge_table[cube_index] == 0) { return; }

    // Find the vertices where the surface intersects the cube
    if (marchingcubes_lookup::edge_table[cube_index] & 1) {
      const Real s = (isolevel - s0) / (s1 - s0);
      vertlist[0]  = p[0] * (1 - s) + p[1] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 2) {
        const Real s      = (isolevel - s1) / (s2 - s1);
        vertlist[1] = p[1] * (1 - s) + p[2] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 4) {
        const Real s      = (isolevel - s2) / (s3 - s2);
        vertlist[2] = p[2] * (1 - s) + p[3] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 8) {
        const Real s      = (isolevel - s3) / (s0 - s3);
        vertlist[3] = p[3] * (1 - s) + p[0] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 16) {
        const Real s      = (isolevel - s4) / (s5 - s4);
        vertlist[4] = p[4] * (1 - s) + p[5] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 32) {
        const Real s      = (isolevel - s5) / (s6 - s5);
        vertlist[5] = p[5] * (1 - s) + p[6] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 64) {
        const Real s      = (isolevel - s6) / (s7 - s6);
        vertlist[6] = p[6] * (1 - s) + p[7] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 128) {
        const Real s      = (isolevel - s7) / (s4 - s7);
        vertlist[7] = p[7] * (1 - s) + p[4] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 256) {
        const Real s      = (isolevel - s0) / (s4 - s0);
        vertlist[8] = p[0] * (1 - s) + p[4] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 512) {
        const Real s      = (isolevel - s1) / (s5 - s1);
        vertlist[9] = p[1] * (1 - s) + p[5] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 1024) {
        const Real s       = (isolevel - s2) / (s6 - s2);
        vertlist[10] = p[2] * (1 - s) + p[6] * s;
      }
      if (marchingcubes_lookup::edge_table[cube_index] & 2048) {
        const Real s       = (isolevel - s3) / (s7 - s3);
        vertlist[11] = p[3] * (1 - s) + p[7] * s;
      }

#ifdef NDEBUG
      {
        std::lock_guard lock{mutex};
#endif
        // create the triangle
        for (size_t i = 0; marchingcubes_lookup::tri_table[cube_index][i] != -1;
             i += 3) {
          iso_volume.insert_face(
              iso_volume.insert_vertex(
                  vertlist[marchingcubes_lookup::tri_table[cube_index][i]]),
              iso_volume.insert_vertex(
                  vertlist[marchingcubes_lookup::tri_table[cube_index][i + 2]]),
              iso_volume.insert_vertex(
                  vertlist[marchingcubes_lookup::tri_table[cube_index]
                                                          [i + 1]]));
        }
#ifdef NDEBUG
      }
#endif
    };
#ifdef NDEBUG
  parallel_nested_for(process_cube, g.size(0) - 1, g.size(1) - 1,
                      g.size(2) - 1);
#else
  nested_for(process_cube, g.size(0) - 1, g.size(1) - 1, g.size(2) - 1);
#endif
  return iso_volume;
}
//------------------------------------------------------------------------------
template <typename Real, typename Indexing, typename BBReal,
          typename IsolevelReal,
          enable_if_arithmetic<Real> = true,
          enable_if_arithmetic<BBReal> = true,
          enable_if_arithmetic<IsolevelReal> = true>
auto marchingcubes(const dynamic_multidim_array<Real, Indexing>& data,
                   const boundingbox<BBReal, 3>& bb, IsolevelReal isolevel) {
  assert(data.num_dimensions() == 3);
  return marchingcubes<Real>(
      [&](auto ix, auto iy, auto iz, const auto & /*ps*/) -> const auto& {
        return data(ix, iy, iz);
      },
      grid{linspace{bb.min(0), bb.max(0), data.size(0)},
           linspace{bb.min(1), bb.max(1), data.size(1)},
           linspace{bb.min(2), bb.max(2), data.size(2)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <typename Real, typename Indexing, typename MemLoc, size_t XRes,
          size_t YRes, size_t ZRes, typename BBReal, typename IsolevelReal,
          enable_if_arithmetic<Real>         = true,
          enable_if_arithmetic<BBReal>       = true,
          enable_if_arithmetic<IsolevelReal> = true>
auto marchingcubes(
    const static_multidim_array<Real, Indexing, MemLoc, XRes, YRes, ZRes>& data,
    const boundingbox<BBReal, 3>& bb, IsolevelReal isolevel) {
  return marchingcubes<Real>(
      [&](auto ix, auto iy, auto iz, const auto & /*ps*/) -> const auto& {
        return data(ix, iy, iz);
      },
      grid{linspace{bb.min(0), bb.max(0), data.size(0)},
           linspace{bb.min(1), bb.max(1), data.size(1)},
           linspace{bb.min(2), bb.max(2), data.size(2)}},
      isolevel);
}
//------------------------------------------------------------------------------
template <typename Field, typename FieldReal, typename GridReal,
          typename IsolevelReal, typename TReal = int>
auto marchingcubes(const field<Field, FieldReal, 3>& sf,
                   const grid<GridReal, 3>& g, IsolevelReal isolevel,
                   TReal t = 0) {
  return marchingcubes<FieldReal>([&](auto /*ix*/, auto /*iy*/, auto /*iz*/,
                                      const auto& pos) { return sf(pos, t); },
                                  g, isolevel);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
