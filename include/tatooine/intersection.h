#ifndef TATOOINE_INTERSECTION_H
#define TATOOINE_INTERSECTION_H

#include <cassert>
#include <optional>
#include "barycentriccoordinates.h"
#include "mesh.h"
#include "ray.h"
#include "tensor.h"
#include "winding_number.h"

//==============================================================================
namespace tatooine::intersection {
//==============================================================================

constexpr double eps = 1e-7;

template <typename Real0, typename Real1>
bool point_in_convex_polygon(
    const pointset<Real0, 2>& ps0, const pointset<Real1, 2>& ps1,
    typename pointset<Real0, 2>::vertex                     p,
    const std::vector<typename pointset<Real1, 2>::vertex>& polygon) {
  return winding_number(ps0, p, ps1, polygon) != 0;
}

//------------------------------------------------------------------------------
template <typename Real>
bool point_in_convex_polygon(
    const pointset<Real, 2>& ps, typename pointset<Real, 2>::vertex p,
    const std::vector<typename pointset<Real, 2>::vertex>& polygon) {
  return winding_number(ps, p, polygon) != 0;
}

//==============================================================================
// BOUNDINGBOX INTERSECTIONS
//==============================================================================
/// point / bounding box
template <size_t N, typename Real, size_t... Is>
constexpr bool point_in_boundingbox(const boundingbox<Real, N>& bb,
                                    const vec<Real, N>&         p,
                                    std::index_sequence<Is...>) {
  static_assert(N == sizeof...(Is),
                "number of indices does not match number of dimensions");
  return ((bb.min(Is) - eps <= p(Is) && p(Is) <= bb.max(Is) + eps) && ...);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// point / bounding box
template <size_t N, typename Real>
constexpr bool point_in_boundingbox(const boundingbox<Real, N>& bb,
                                    const vec<Real, N>&         p) {
  return point_in_boundingbox(bb, p, std::make_index_sequence<N>{});
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// point / bounding box
template <size_t N, typename Real>
constexpr bool point_in_boundingbox(const vec<Real, N>&         p,
                                    const boundingbox<Real, N>& bb) {
  return point_in_boundingbox(bb, p);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// point / bounding box
template <size_t N, typename Real>
constexpr bool point_in_boundingbox(const pointset<Real, N>&           ps,
                                    typename pointset<Real, N>::vertex v,
                                    const boundingbox<Real, N>&        bb) {
  return point_in_boundingbox(bb, ps[v]);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// point / bounding box
template <size_t N, typename Real>
constexpr bool point_in_boundingbox(const boundingbox<Real, N>&        bb,
                                    const pointset<Real, N>&           ps,
                                    typename pointset<Real, N>::vertex v) {
  return point_in_boundingbox(bb, ps[v]);
}

//------------------------------------------------------------------------------
/// 2d ray / 2d ray
template <typename Real>
constexpr std::optional<std::pair<Real, Real>> ray_hits_ray(
    const ray<Real, 2>& r0, const ray<Real, 2>& r1) {
  if (approx_equal(r0.direction, r1.direction, eps)) return {};

  Real a = 1 / (-r1.direction(0) * r0.direction(1) +
                r0.direction(0) * r1.direction(1));
  Real s = (r1.direction(0) * (r0.origin(1) - r1.origin(1)) -
            r1.direction(1) * (r0.origin(0) - r1.origin(0))) *
           a;
  Real t = (-r0.direction(1) * (r0.origin(0) - r1.origin(0)) +
            r0.direction(0) * (r0.origin(1) - r1.origin(1))) *
           a;
  return std::pair{s, t};
}

//----------------------------------------------------------------------------
/// 2d edge / 2d edge
template <typename Real>
std::optional<vec<Real, 2>> edge_hits_edge(const edgeset<Real, 2>&         es0,
                                           typename edgeset<Real, 2>::edge e0,
                                           const edgeset<Real, 2>&         es1,
                                           typename edgeset<Real, 2>::edge e1) {
  ray r0{es0[e0][0], es0[e0][1] - es0[e0][0]};
  ray r1{es1[e1][0], es1[e1][1] - es1[e1][0]};
  if (auto i = ray_hits_ray(r0, r1);
      i && -eps <= i->first && i->first <= 1 + eps)
    return r0->at(i->first);
  return {};
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// 2d edge / 2d edge
template <typename Real>
auto edge_hits_edge(const edgeset<Real, 2>&         es,
                    typename edgeset<Real, 2>::edge e0,
                    typename edgeset<Real, 2>::edge e1) {
  return edge_hits_edge(es, e0, es, e1);
}

//----------------------------------------------------------------------------
/// 2d triangle / 2d triangle
template <typename Real>
constexpr bool triangle_in_triangle(
    const vec<Real, 2>& p00, const vec<Real, 2>& p01, const vec<Real, 2>& p02,
    const vec<Real, 2>& p10, const vec<Real, 2>& p11, const vec<Real, 2>& p12) {
  auto s00_01 = p01 - p00;
  auto s00_02 = p02 - p00;
  auto s01_02 = p02 - p01;
  auto s10_11 = p11 - p10;
  auto s10_12 = p12 - p10;
  auto s11_12 = p12 - p11;

  return ray_ray({p00, s00_01}, {p10, s10_11}) ||
         ray_ray({p00, s00_01}, {p10, s10_12}) ||
         ray_ray({p00, s00_01}, {p11, s11_12}) ||

         ray_ray({p00, s00_02}, {p10, s10_11}) ||
         ray_ray({p00, s00_02}, {p10, s10_12}) ||
         ray_ray({p00, s00_02}, {p11, s11_12}) ||

         ray_ray({p01, s01_02}, {p10, s10_11}) ||
         ray_ray({p01, s01_02}, {p10, s10_12}) ||
         ray_ray({p01, s01_02}, {p11, s11_12});
}

//----------------------------------------------------------------------------
/// 2d triangle / 2d triangle
template <typename Real, size_t M, size_t N>
constexpr bool triangle_in_triangle(const mesh<Real, M>&         m0,
                                    typename mesh<Real, M>::face f0,
                                    const mesh<Real, N>&         m1,
                                    typename mesh<Real, N>::face f1) {
  assert(m0.num_vertices(f0) == 3 && m1.num_vertices(f1) == 3);
  return triangle_in_triangle(m0[m0[f0][0]], m0[m0[f0][1]], m0[m0[f0][2]],
                              m1[m1[f1][0]], m1[m1[f1][1]], m1[m1[f1][2]]);
}

//----------------------------------------------------------------------------
/// 2d triangle / 2d triangle
template <typename Real, size_t N>
constexpr bool triangle_in_triangle(const mesh<Real, N>&         m,
                                    typename mesh<Real, N>::face f0,
                                    typename mesh<Real, N>::face f1) {
  assert(m.num_vertices(f0) == 3);
  return triangle_in_triangle(m[m[f0][0]], m[m[f0][1]], m[m[f0][2]],
                              m[m[f1][0]], m[m[f1][1]], m[m[f1][2]]);
}

//------------------------------------------------------------------------------
/// ray / bounding box
template <typename Real, size_t N>
std::optional<std::pair<Real, vec<Real, N>>> ray_hits_boundingbox(
    const ray<Real, N>& r, const boundingbox<Real, N>& bb) {
  enum pos : unsigned short { right, left, middle };

  std::array<pos, N> quadrant;
  vec<Real, N>       candidate_plane;

  // find candidate planes
  bool inside = true;
  for (size_t i = 0; i < N; ++i)
    if (r.origin[i] < bb.min[i]) {
      quadrant[i]        = left;
      candidate_plane[i] = bb.min[i];
      inside             = false;
    } else if (r.origin[i] > bb.max[i]) {
      quadrant[i]        = right;
      candidate_plane[i] = bb.max[i];
      inside             = false;
    } else
      quadrant[i] = middle;

  // ray origin inside bounding box
  if (inside) return std::pair{Real(0), r.origin};

  // calculate t distances to candidate planes
  vec<Real, N> max_t;
  for (size_t i = 0; i < N; ++i)
    if (quadrant[i] != middle && std::abs(r.direction[i]) > eps)
      max_t(i) = (candidate_plane[i] - r.origin[i]) / r.direction[i];
    else
      max_t(i) = -1;

  auto largest_max_t = boost::max_element(max_t);

  // check final candidate actually inside box
  if (*largest_max_t < -eps) return {};
  vec<Real, N> hit_coord;
  size_t       i = 0;
  for (auto it = begin(max_t); it != end(max_t); ++it, ++i)
    if (largest_max_t != it) {
      hit_coord[i] = r.origin[i] + *largest_max_t * r.direction[i];
      if (bb.min[i] > hit_coord[i] || hit_coord[i] > bb.max[i]) return {};
    } else
      hit_coord[i] = candidate_plane[i];
  return std::pair{*largest_max_t, hit_coord};
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// ray / bounding box
template <typename Real, size_t N>
auto ray_hits_boundingbox(const boundingbox<Real, N>& bb,
                          const ray<Real, N>&         r) {
  return ray_hits_boundingbox(r, bb);
}
//------------------------------------------------------------------------------
/// edge / bounding box
template <typename Real, size_t N>
bool edge_in_boundingbox(const boundingbox<Real, N>& bb, const vec<Real, N>& v0,
                         const vec<Real, N>& v1) {
  auto hit = ray_hits_boundingbox(ray{v0, v1 - v0}, bb);
  if (!hit) return false;
  return hit->first <= 1 + eps;
}
//------------------------------------------------------------------------------
/// edge / bounding box
template <typename Real, size_t N>
bool edge_in_boundingbox(const edgeset<Real, N>&         es,
                         typename edgeset<Real, N>::edge e,
                         const boundingbox<Real, N>&     bb) {
  return edge_in_boundingbox(bb, es[es[e][0]], es[es[e][1]]);
}

//==============================================================================
// MESH INTERSECTIONS
//==============================================================================
/// 2d point / 2d triangle
template <typename Real, size_t... Is>
std::optional<vec<Real, 3>> point_in_triangle(const vec<Real, 2>& p,
                                              const vec<Real, 2>& v0,
                                              const vec<Real, 2>& v1,
                                              const vec<Real, 2>& v2) {
  auto b = barycentric_coordinates(p, v0, v1, v2);
  if (b(0) >= -eps && b(1) >= -eps && b(2) >= -eps) return b;
  return {};
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, size_t... Is>
std::optional<vec<Real, 3>> point_in_triangle(const vec<Real, 2>&          p,
                                              const mesh<Real, 2>&         m,
                                              typename mesh<Real, 2>::face f,
                                              std::index_sequence<Is...>) {
  assert(m.num_vertices(f) == 3);
  static_assert(sizeof...(Is) == 2);
  auto b = barycentric_coordinates(m, f, p);
  if (((b(Is) >= -eps) && ...)) return b;
  return {};
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// 2d point / 2d triangle
template <typename Real>
auto point_in_triangle(const vec<Real, 2>& p, const mesh<Real, 2>& m,
                       typename mesh<Real, 2>::face f) {
  return point_in_triangle(p, m, f, std::make_index_sequence<2>{});
}

//------------------------------------------------------------------------------
/// Möller–Trumbore intersection algorithm
/// 3d ray / 3d triangle
template <typename Real>
std::optional<std::pair<Real, vec<Real, 3>>> ray_hits_triangle(
    const ray<Real, 3>& r, const vec<Real, 3>& v0, const vec<Real, 3>& v1,
    const vec<Real, 3>& v2) {
  const auto edge1 = v1 - v0;
  const auto edge2 = v2 - v0;
  auto       h     = cross(r.direction, edge2);
  auto       a     = dot(edge1, h);
  // ray is parallel to triangle.
  if (-eps < a && a < eps) return {};
  auto f = 1 / a;
  auto s = r.origin - v0;
  auto u = f * dot(s, h);
  if (u < -eps || u > 1 + eps) return {};
  auto q = cross(s, edge1);
  auto v = f * dot(r.direction, q);
  if (v < -eps || u + v > 1 + eps) return {};
  // At this stage we can compute t to find out where the intersection point
  // is on the line.
  Real t = f * dot(edge2, q);
  // ray intersection
  return std::pair{t, vec{u, v, 1 - u - v}};
}

//------------------------------------------------------------------------------
/// 3d ray / 3d triangle
template <typename Real>
auto ray_hits_triangle(const ray<Real, 3>& r, const mesh<Real, 3>& m,
                       typename mesh<Real, 3>::face f) {
  return ray_hits_triangle(r, m[m[f][0]], m[m[f][1]], m[m[f][2]]);
}

//------------------------------------------------------------------------------
template <typename Real>
std::optional<Real> ray_hits_tetraeder(const ray<Real, 3>& r,
                                       const vec<Real, 3>& v0,
                                       const vec<Real, 3>& v1,
                                       const vec<Real, 3>& v2,
                                       const vec<Real, 3>& v3) {
  return ray_hits_triangle(r, v0, v1, v2) || ray_hits_triangle(r, v0, v1, v3) ||
         ray_hits_triangle(r, v0, v2, v3) || ray_hits_triangle(r, v1, v2, v3);
}

//------------------------------------------------------------------------------
/// 3d ray / 3d tetraeder
template <typename Real>
auto ray_hits_triangle(const ray<Real, 3>& r, const tetraedermesh<Real, 3>& m,
                       typename tetraedermesh<Real, 3>::tetraeder t) {
  return ray_hits_tetraeder(r, m[m[t][0]], m[m[t][1]], m[m[t][2]], m[m[t][3]]);
}

//------------------------------------------------------------------------------
/// 3d edge / 3d triangle
template <typename Real>
std::optional<vec<Real, 3>> edge_hits_triangle(
    const edgeset<Real, 3>& es, typename edgeset<Real, 3>::edge e,
    const mesh<Real, 3>& m, typename mesh<Real, 3>::face f) {
  assert(m.num_vertices(f) == 3);
  ray r{es[e][0], es[e][1] - es[e][0]};
  if (auto hit = ray_hits_triangle(r, m, f);
      hit->first && -eps <= hit->first && hit->first <= 1 + eps)
    return r(hit->first);
  return {};
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// 3d edge / 3d triangle
template <typename Real>
auto edge_hits_triangle(const mesh<Real, 3>& m, typename mesh<Real, 3>::edge e,
                        typename mesh<Real, 3>::face f) {
  return edge_hits_triangle(m, e, m, f);
}

//==============================================================================
/// 3d triangle / 3d bounding box
template <typename Real>
bool triangle_in_boundingbox(const boundingbox<Real, 3>& bb,
                             const vec<Real, 3>& v0, const vec<Real, 3>& v1,
                             const vec<Real, 3>& v2) {
  // check if one of the edges is in boundary
  if (edge_in_boundingbox(bb, v0, v1) || edge_in_boundingbox(bb, v0, v2) ||
      edge_in_boundingbox(bb, v1, v2))
    return true;

  // check if any boundary edge hits triangle
  return ray_hits_triangle({vec{bb.min[0], bb.min[1], bb.min[2]},
                            vec{bb.max[0], bb.min[1], bb.min[2]} -
                                vec{bb.min[0], bb.min[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.min[0], bb.min[1], bb.min[2]},
                            vec{bb.min[0], bb.max[1], bb.min[2]} -
                                vec{bb.min[0], bb.min[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.min[0], bb.min[1], bb.min[2]},
                            vec{bb.min[0], bb.min[1], bb.max[2]} -
                                vec{bb.min[0], bb.min[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.max[0], bb.min[1], bb.min[2]},
                            vec{bb.max[0], bb.max[1], bb.min[2]} -
                                vec{bb.max[0], bb.min[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.max[0], bb.min[1], bb.min[2]},
                            vec{bb.max[0], bb.min[1], bb.max[2]} -
                                vec{bb.max[0], bb.min[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.min[0], bb.max[1], bb.min[2]},
                            vec{bb.max[0], bb.min[1], bb.max[2]} -
                                vec{bb.min[0], bb.max[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.min[0], bb.max[1], bb.min[2]},
                            vec{bb.min[0], bb.max[1], bb.max[2]} -
                                vec{bb.min[0], bb.max[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.max[0], bb.min[1], bb.max[2]},
                            vec{bb.max[0], bb.max[1], bb.max[2]} -
                                vec{bb.max[0], bb.min[1], bb.max[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.min[0], bb.max[1], bb.min[2]},
                            vec{bb.max[0], bb.max[1], bb.min[2]} -
                                vec{bb.min[0], bb.max[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.min[0], bb.max[1], bb.min[2]},
                            vec{bb.min[0], bb.max[1], bb.max[2]} -
                                vec{bb.min[0], bb.max[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.max[0], bb.max[1], bb.min[2]},
                            vec{bb.max[0], bb.max[1], bb.max[2]} -
                                vec{bb.max[0], bb.max[1], bb.min[2]}},
                           v0, v1, v2) ||
         ray_hits_triangle({vec{bb.min[0], bb.max[1], bb.max[2]},
                            vec{bb.max[0], bb.max[1], bb.max[2]} -
                                vec{bb.min[0], bb.max[1], bb.max[2]}},
                           v0, v1, v2);
}

//------------------------------------------------------------------------------
template <typename Real>
bool triangle_in_boundingbox(const mesh<Real, 3>&         m,
                             typename mesh<Real, 3>::face f,
                             const boundingbox<Real, 3>&  bb) {
  return triangle_in_boundingbox(bb, m[m[f][0]], m[m[f][1]], m[m[f][2]]);
}

//------------------------------------------------------------------------------
/// 2d triangle / 2d bounding box
template <typename Real>
bool triangle_in_boundingbox(const mesh<Real, 2>&         m,
                             typename mesh<Real, 2>::face f,
                             const boundingbox<Real, 2>&  bb) {
  // check if one of the edges is in boundary
  for (const auto e : m.edges(f))
    if (edge_in_boundingbox(m, e, bb)) return true;

  // check if any boundary corner is in the triangle
  return point_in_triangle(bb.min, m, f) || point_in_triangle(bb.max, m, f) ||
         point_in_triangle(vec{bb.min(0), bb.max(1)}, m, f) ||
         point_in_triangle(vec{bb.max(0), bb.min(1)}, m, f);
}

//------------------------------------------------------------------------------
/// 3d point / 3d tetraeder
template <typename Real>
std::optional<vec<Real, 4>> point_in_tetraeder(const vec<Real, 3>& p,
                                               const vec<Real, 3>& v0,
                                               const vec<Real, 3>& v1,
                                               const vec<Real, 3>& v2,
                                               const vec<Real, 3>& v3) {
  auto b = barycentric_coordinates(p, v0, v1, v2, v3);
  if (b(0) >= -eps && b(1) >= -eps && b(2) >= -eps && b(3) >= -eps) return b;
  return {};
}
//------------------------------------------------------------------------------
/// 3d point / 3d tetraeder
template <typename Real>
auto point_in_tetraeder(const tetraedermesh<Real, 3>&              m,
                        typename tetraedermesh<Real, 3>::tetraeder t,
                        const vec<Real, 3>&                        p) {
  return point_in_tetraeder(p, m[m[t][0]], m[m[t][1]], m[m[t][2]], m[m[t][3]]);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// 3d point / 3d tetraeder
template <typename Real>
auto point_in_tetraeder(const vec<Real, 3>& p, const tetraedermesh<Real, 3>& m,
                        typename tetraedermesh<Real, 3>::tetraeder t) {
  return point_in_tetraeder(p, m[m[t][0]], m[m[t][1]], m[m[t][2]], m[m[t][3]]);
}

//==============================================================================
template <typename Real>
bool tetraeder_in_boundingbox(const boundingbox<Real, 3>& bb,
                              const vec<Real, 3>& v0, const vec<Real, 3>& v1,
                              const vec<Real, 3>& v2, const vec<Real, 3>& v3) {
  return point_in_boundingbox(bb, v0) || point_in_boundingbox(bb, v1) ||
         point_in_boundingbox(bb, v2) || point_in_boundingbox(bb, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.min[0], bb.min[1], bb.min[2]}, v0,
                            v1, v2, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.min[0], bb.min[1], bb.max[2]}, v0,
                            v1, v2, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.min[0], bb.max[1], bb.min[2]}, v0,
                            v1, v2, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.min[0], bb.max[1], bb.max[2]}, v0,
                            v1, v2, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.max[0], bb.min[1], bb.min[2]}, v0,
                            v1, v2, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.max[0], bb.min[1], bb.max[2]}, v0,
                            v1, v2, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.max[0], bb.max[1], bb.min[2]}, v0,
                            v1, v2, v3) ||
         point_in_tetraeder(vec<Real, 3>{bb.max[0], bb.max[1], bb.max[2]}, v0,
                            v1, v2, v3) ||
         triangle_in_boundingbox(bb, v0, v1, v2) ||
         triangle_in_boundingbox(bb, v0, v1, v3) ||
         triangle_in_boundingbox(bb, v0, v2, v3) ||
         triangle_in_boundingbox(bb, v1, v2, v3);
}
//------------------------------------------------------------------------------
/// 3d tetraeder / 3d boundingbox
template <typename Real>
bool tetraeder_in_boundingbox(const tetraedermesh<Real, 3>&              m,
                              typename tetraedermesh<Real, 3>::tetraeder t,
                              const boundingbox<Real, 3>&                bb) {
  return tetraeder_in_boundingbox(bb, m[m[t][0]], m[m[t][1]], m[m[t][2]],
                                  m[m[t][3]]);
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// 3d tetraeder / 3d boundingbox
template <typename Real>
bool tetraeder_in_boundingbox(const boundingbox<Real, 3>&                bb,
                              const tetraedermesh<Real, 3>&              m,
                              typename tetraedermesh<Real, 3>::tetraeder t) {
  return tetraeder_in_boundingbox(bb, m[m[t][0], m[m[t][1], m[m[t][2]]]],
                                  m[m[t][3]]);
}

//==============================================================================
}  // namespace tatooine::intersection
//==============================================================================

#endif
