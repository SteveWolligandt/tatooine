#ifndef TATOOINE_MARCHING_CUBES_H
#define TATOOINE_MARCHING_CUBES_H

#include <omp.h>

#include <string>
#include <vector>

#include "field.h"
#include "hdf5.h"
#include "marchingcubeslookuptable.h"
#include "simple_tri_mesh.h"
#include "tensor.h"
#include "utility.h"

//==============================================================================
namespace tatooine {
//==============================================================================
/// \brief      Indexing and lookup map from
/// http://paulbourke.net/geometry/polygonise/
template <typename Field, typename FieldReal, typename GridReal,
          typename IsolevelReal, typename TReal = double>
auto marchingcubes(const field<Field, FieldReal, 3>& sf,
                   const grid<GridReal, 3>& g, IsolevelReal isolevel,
                   TReal t = 0) {
  using real_t = FieldReal;
  using pos_t = typename Field::pos_t;
  simple_tri_mesh<FieldReal, 3> iso_volume;

  const auto  spacing = g.spacing();
  const auto& min     = g.min();

#ifdef NDEBUG
    omp_lock_t writelock;
    omp_init_lock(&writelock);
#pragma omp parallel for collapse(3)
#endif
    for (size_t z = 0; z < g.dimension(2).size() - 1; z++) {
      for (size_t y = 0; y < g.dimension(1).size() - 1; y++) {
        for (size_t x = 0; x < g.dimension(0).size() - 1; x++) {
          FieldReal  s0, s1, s2, s3, s4, s5, s6, s7;
          auto  vertlist = make_array<pos_t, 12>(pos_t::zeros());
          pos_t p0 = min +
                     vec3{spacing(0) * (x),
                          spacing(1) * (y),
                          spacing(2) * (z + 1)};
          pos_t p1 = min +
                     vec3{spacing(0) * (x + 1),
                          spacing(1) * (y),
                          spacing(2) * (z + 1)};
          pos_t p2 = min +
                     vec3{spacing(0) * (x + 1),
                          spacing(1) * (y),
                          spacing(2) * (z)};
          pos_t p3 = min +
                     vec3{spacing(0) * (x),
                          spacing(1) * (y),
                          spacing(2) * (z)};
          pos_t p4 = min +
                     vec3{spacing(0) * (x),
                          spacing(1) * (y + 1),
                          spacing(2) * (z + 1)};
          pos_t p5 = min +
                     vec3{spacing(0) * (x + 1),
                          spacing(1) * (y + 1),
                          spacing(2) * (z + 1)};
          pos_t p6 = min +
                     vec3{spacing(0) * (x + 1),
                          spacing(1) * (y + 1),
                          spacing(2) * (z)};
          pos_t p7 = min +
                     vec3{spacing(0) * (x),
                          spacing(1) * (y + 1),
                          spacing(2) * (z)};

          s0 = sf(p0, t);
          s1 = sf(p1, t);
          s2 = sf(p2, t);
          s3 = sf(p3, t);
          s4 = sf(p4, t);
          s5 = sf(p5, t);
          s6 = sf(p6, t);
          s7 = sf(p7, t);

          int cube_index = 0;
          if (s0 < isolevel) { cube_index |= 1; }
          if (s1 < isolevel) { cube_index |= 2; }
          if (s2 < isolevel) { cube_index |= 4; }
          if (s3 < isolevel) { cube_index |= 8; }
          if (s4 < isolevel) { cube_index |= 16; }
          if (s5 < isolevel) { cube_index |= 32; }
          if (s6 < isolevel) { cube_index |= 64; }
          if (s7 < isolevel) { cube_index |= 128; }

          // Cube is entirely in/out of the surface
          if (marchingcubes_lookup::edgeTable[cube_index] == 0) { continue; }

          /* Find the vertices where the surface intersects the cube */
          if (marchingcubes_lookup::edgeTable[cube_index] & 1) {
            real_t s      = (isolevel - s0) / (s1 - s0);
            vertlist[0] = p0 * (1 - s) + p1 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 2) {
            real_t s      = (isolevel - s1) / (s2 - s1);
            vertlist[1] = p1 * (1 - s) + p2 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 4) {
            real_t s      = (isolevel - s2) / (s3 - s2);
            vertlist[2] = p2 * (1 - s) + p3 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 8) {
            real_t s      = (isolevel - s3) / (s0 - s3);
            vertlist[3] = p3 * (1 - s) + p0 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 16) {
            real_t s      = (isolevel - s4) / (s5 - s4);
            vertlist[4] = p4 * (1 - s) + p5 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 32) {
            real_t s      = (isolevel - s5) / (s6 - s5);
            vertlist[5] = p5 * (1 - s) + p6 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 64) {
            real_t s      = (isolevel - s6) / (s7 - s6);
            vertlist[6] = p6 * (1 - s) + p7 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 128) {
            real_t s      = (isolevel - s7) / (s4 - s7);
            vertlist[7] = p7 * (1 - s) + p4 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 256) {
            real_t s      = (isolevel - s0) / (s4 - s0);
            vertlist[8] = p0 * (1 - s) + p4 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 512) {
            real_t s      = (isolevel - s1) / (s5 - s1);
            vertlist[9] = p1 * (1 - s) + p5 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 1024) {
            real_t s       = (isolevel - s2) / (s6 - s2);
            vertlist[10] = p2 * (1 - s) + p6 * s;
          }
          if (marchingcubes_lookup::edgeTable[cube_index] & 2048) {
            real_t s       = (isolevel - s3) / (s7 - s3);
            vertlist[11] = p3 * (1 - s) + p7 * s;
          }

#ifdef NDEBUG
          omp_set_lock(&writelock);
#endif
          // create the triangle
          for (size_t i = 0;
               marchingcubes_lookup::triTable[cube_index][i] != -1; i += 3) {
            iso_volume.insert_face(
                iso_volume.insert_vertex(
                    vertlist[marchingcubes_lookup::triTable[cube_index][i]]),
                iso_volume.insert_vertex(
                    vertlist[marchingcubes_lookup::triTable[cube_index]
                                                           [i + 2]]),
                iso_volume.insert_vertex(
                    vertlist[marchingcubes_lookup::triTable[cube_index]
                                                           [i + 1]]));
          }
#ifdef NDEBUG
          omp_unset_lock(&writelock);
#endif
        }
      }
    }

    return iso_volume;
}
//------------------------------------------------------------------------------
template <typename Real, typename MinReal, typename IsolevelReal>
auto marchingcubes(const h5::multi_array<Real>& data,
                   const vec<MinReal, 3>& min, const vec<MinReal, 3>& spacing,
                   IsolevelReal isolevel) {
  assert(data.num_dimensions() == 3);
  using pos_t = vec<Real, 3>;
  simple_tri_mesh<Real, 3> iso_volume;

#ifdef NDEBUG
  omp_lock_t writelock;
  omp_init_lock(&writelock);
#pragma omp parallel for collapse(3)
#endif
  for (size_t z = 0; z < data.resolution(2) - 1; z++) {
    for (size_t y = 0; y < data.resolution(1) - 1; y++) {
      for (size_t x = 0; x < data.resolution(0) - 1; x++) {
        auto  vertlist = make_array<pos_t, 12>(pos_t::zeros());
        pos_t p0 = min +
                   vec3{spacing(0) * (x),
                        spacing(1) * (y),
                        spacing(2) * (z + 1)};
        pos_t p1 = min +
                   vec3{spacing(0) * (x + 1),
                        spacing(1) * (y),
                        spacing(2) * (z + 1)};
        pos_t p2 = min +
                   vec3{spacing(0) * (x + 1),
                        spacing(1) * (y),
                        spacing(2) * (z)};
        pos_t p3 = min +
                   vec3{spacing(0) * (x),
                        spacing(1) * (y),
                        spacing(2) * (z)};
        pos_t p4 = min +
                   vec3{spacing(0) * (x),
                        spacing(1) * (y + 1),
                        spacing(2) * (z + 1)};
        pos_t p5 = min +
                   vec3{spacing(0) * (x + 1),
                        spacing(1) * (y + 1),
                        spacing(2) * (z + 1)};
        pos_t p6 = min +
                   vec3{spacing(0) * (x + 1),
                        spacing(1) * (y + 1),
                        spacing(2) * (z)};
        pos_t p7 = min +
                   vec3{spacing(0) * (x),
                        spacing(1) * (y + 1),
                        spacing(2) * (z)};

        const auto& s0 = data(x,     y,     z + 1);
        const auto& s1 = data(x + 1, y,     z + 1);
        const auto& s2 = data(x + 1, y,     z);
        const auto& s3 = data(x,     y,     z);
        const auto& s4 = data(x,     y + 1, z + 1);
        const auto& s5 = data(x + 1, y + 1, z + 1);
        const auto& s6 = data(x + 1, y + 1, z);
        const auto& s7 = data(x,     y + 1, z);

        int cube_index = 0;
        if (s0 < isolevel) { cube_index |= 1; }
        if (s1 < isolevel) { cube_index |= 2; }
        if (s2 < isolevel) { cube_index |= 4; }
        if (s3 < isolevel) { cube_index |= 8; }
        if (s4 < isolevel) { cube_index |= 16; }
        if (s5 < isolevel) { cube_index |= 32; }
        if (s6 < isolevel) { cube_index |= 64; }
        if (s7 < isolevel) { cube_index |= 128; }

        // Cube is entirely in/out of the surface
        if (marchingcubes_lookup::edgeTable[cube_index] == 0) { continue; }

        // Find the vertices where the surface intersects the cube
        if (marchingcubes_lookup::edgeTable[cube_index] & 1) {
          Real s      = (isolevel - s0) / (s1 - s0);
          vertlist[0] = p0 * (1 - s) + p1 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 2) {
          Real s      = (isolevel - s1) / (s2 - s1);
          vertlist[1] = p1 * (1 - s) + p2 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 4) {
          Real s      = (isolevel - s2) / (s3 - s2);
          vertlist[2] = p2 * (1 - s) + p3 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 8) {
          Real s      = (isolevel - s3) / (s0 - s3);
          vertlist[3] = p3 * (1 - s) + p0 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 16) {
          Real s      = (isolevel - s4) / (s5 - s4);
          vertlist[4] = p4 * (1 - s) + p5 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 32) {
          Real s      = (isolevel - s5) / (s6 - s5);
          vertlist[5] = p5 * (1 - s) + p6 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 64) {
          Real s      = (isolevel - s6) / (s7 - s6);
          vertlist[6] = p6 * (1 - s) + p7 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 128) {
          Real s      = (isolevel - s7) / (s4 - s7);
          vertlist[7] = p7 * (1 - s) + p4 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 256) {
          Real s      = (isolevel - s0) / (s4 - s0);
          vertlist[8] = p0 * (1 - s) + p4 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 512) {
          Real s      = (isolevel - s1) / (s5 - s1);
          vertlist[9] = p1 * (1 - s) + p5 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 1024) {
          Real s       = (isolevel - s2) / (s6 - s2);
          vertlist[10] = p2 * (1 - s) + p6 * s;
        }
        if (marchingcubes_lookup::edgeTable[cube_index] & 2048) {
          Real s       = (isolevel - s3) / (s7 - s3);
          vertlist[11] = p3 * (1 - s) + p7 * s;
        }

#ifdef NDEBUG
        omp_set_lock(&writelock);
#endif
        // create the triangle
        for (size_t i = 0; marchingcubes_lookup::triTable[cube_index][i] != -1;
             i += 3) {
          iso_volume.insert_face(
              iso_volume.insert_vertex(
                  vertlist[marchingcubes_lookup::triTable[cube_index][i]]),
              iso_volume.insert_vertex(
                  vertlist[marchingcubes_lookup::triTable[cube_index][i + 2]]),
              iso_volume.insert_vertex(
                  vertlist[marchingcubes_lookup::triTable[cube_index][i + 1]]));
        }
#ifdef NDEBUG
        omp_unset_lock(&writelock);
#endif
      }
    }
  }

  return iso_volume;
}
//------------------------------------------------------------------------------
template <typename Real, typename BBReal, typename IsolevelReal>
auto marchingcubes(const h5::multi_array<Real>&  data,
                   const boundingbox<BBReal, 3>& bb, IsolevelReal isolevel) {
  assert(data.num_dimensions() == 3);
  return marchingcubes(data, bb.min,
                       vec{(bb.max(0) - bb.min(0)) / (data.resolution(0) - 1),
                           (bb.max(1) - bb.min(1)) / (data.resolution(1) - 1),
                           (bb.max(2) - bb.min(2)) / (data.resolution(2) - 1)},
                       isolevel);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
