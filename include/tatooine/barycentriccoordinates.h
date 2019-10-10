#ifndef TATOOINE_BARYCENTRIC_COORDINATES_H
#define TATOOINE_BARYCENTRIC_COORDINATES_H

#include "mesh.h"
#include "tetraedermesh.h"

//==============================================================================
namespace tatooine {
//==============================================================================
/// barycentric coordinates of a point p in triangle [v0,v1,v2]
template <typename Real>
auto barycentric_coordinates(const vec<Real, 2>& p, const vec<Real, 2>& v0,
                             const vec<Real, 2>& v1, const vec<Real, 2>& v2) {
  auto d0    = v1 - v0;
  auto d1    = v2 - v0;
  auto d2    = p - v0;
  auto d00   = dot(d0, d0);
  auto d01   = dot(d0, d1);
  auto d11   = dot(d1, d1);
  auto d20   = dot(d2, d0);
  auto d21   = dot(d2, d1);
  auto denom = 1 / (d00 * d11 - d01 * d01);
  vec  b{1, (d11 * d20 - d01 * d21) * denom, (d00 * d21 - d01 * d20) * denom};
  b(0) -= b(1);
  b(0) -= b(2);
  return b;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// barycentric coordinates of a point p in triangle f of mesh m
template <typename Real>
auto barycentric_coordinates(const mesh<Real, 2>&         m,
                             typename mesh<Real, 2>::face f,
                             const vec<Real, 2>&          p) {
  assert(m.num_vertices(f) == 3);
  return barycentric_coordinates(p, m[m[f][0]], m[m[f][1]], m[m[f][2]]);
}

//------------------------------------------------------------------------------
/// barycentric coordinates of point p in tetraeder [t0,t1,t2,t3]
template <typename Real>
auto barycentric_coordinates(const vec<Real, 3>& p, const vec<Real, 3>& t0,
                             const vec<Real, 3>& t1, const vec<Real, 3>& t2,
                             const vec<Real, 3>& t3) {
  auto ap = p - t0;
  auto bp = p - t1;
  auto ab = t1 - t0;
  auto ac = t2 - t0;
  auto ad = t3 - t0;
  auto bc = t2 - t1;
  auto bd = t3 - t1;

  auto v = 1 / dot(ab, cross(ac, ad));
  return vec{dot(bp, cross(bd, bc)) * v, dot(ap, cross(ac, ad)) * v,
             dot(ap, cross(ad, ab)) * v, dot(ap, cross(ab, ac)) * v};
}

//------------------------------------------------------------------------------
/// barycentric coordinates of point p in tetraeder t of tet mesh m
template <typename Real>
auto barycentric_coordinates(const tetraedermesh<Real, 3>&              m,
                             typename tetraedermesh<Real, 3>::tetraeder t,
                             const vec<Real, 3>&                        p) {
  return barycentric_coordinates(p, m[m[t][0], m[m[t][1], m[m[t][2]]]],
                                 m[m[t][3]]);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
