#ifndef TATOOINE_GEOMETRY_SPHERE_RAY_INTERSECTION_H
#define TATOOINE_GEOMETRY_SPHERE_RAY_INTERSECTION_H
//==============================================================================
#include <tatooine/ray.h>
#include <tatooine/polynomial.h>
#include <tatooine/intersection.h>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <typename Real, size_t N> struct sphere;
//==============================================================================
template <typename Real>
std::optional<intersection<Real, 3>> check_intersection(const ray<Real, 3>&    r,
                                               const sphere<Real, 3>& s,
                                               const Real min_t = 0) {
  const auto  L          = r.origin() - s.center();
  const Real  c          = dot(r.direction(), r.direction());
  const Real  b          = 2 * dot(r.direction(), L);
  const Real  a          = dot(L, L) - s.radius() * s.radius();
  auto        solutions  = solve(polynomial{a, b, c});
  if (solutions.empty()) { return {}; }

  Real t = 0;
  if (size(solutions) == 1) {
    if (t < 0 || t < min_t) {
      return {};
    };
    t = solutions[0];
  }

  if (solutions[1] < 0) {
    t = solutions[0];
  } else if (solutions[0] < 0) {
    t = solutions[1];
  } else if (solutions[0] < solutions[1]) {
    t = solutions[0] > min_t ? solutions[0] : solutions[1];
  } else {
    t = solutions[1] > min_t ? solutions[1] : solutions[0];
  }

  auto hit_pos = r(t);
  auto nor     = normalize(hit_pos - s.center());
  vec  uv{std::atan2(nor(0), nor(2)) / (2 * M_PI) + M_PI / 2,
         std::acos(-nor(1)) / M_PI};
  return intersection<Real, 3>{&s, r, t, hit_pos, nor, uv};
}
//------------------------------------------------------------------------------
template <typename Real, size_t N>
std::optional<intersection<Real, N>> check_intersection(const sphere<Real, N>& s,
                                               const ray<Real, N>&    r,
                                               const Real min_t = 0) {
  return check_intersection(r,s, min_t);
}
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif
