#ifndef TATOOINE_GEOMETRY_SPHERE_RAY_INTERSECTION_H
#define TATOOINE_GEOMETRY_SPHERE_RAY_INTERSECTION_H
//==============================================================================
#include <tatooine/intersection.h>
#include <tatooine/polynomial.h>
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <floating_point Real, size_t N>
struct sphere;
//==============================================================================
template <floating_point Real>
std::optional<intersection<Real>> check_intersection(
    ray<Real, 3> const& r, sphere<Real, 3> const& s, Real const min_t = 0) {
  auto const m = r.origin() - s.center();
  auto const b = dot(m, r.direction());
  auto const c = dot(m, m) - s.radius() * s.radius();

  // Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0)
  if (c > 0 && b > 0) {
    return {};
  }
  auto const discr = b * b - c;

  // A negative discriminant corresponds to ray missing sphere
  if (discr < 0) {
    return {};
  }

  // Ray now found to intersect sphere, compute smallest t value of intersection
  auto t = -b - std::sqrt(discr);

  // If t is negative, ray started inside sphere so clamp t to zero
  if (t < min_t) {
    return {};
  }

  auto const hit_pos = r(t);
  auto const nor     = normalize(hit_pos - s.center());
  vec        uv{std::atan2(nor(0), nor(2)) / (2 * M_PI) + M_PI / 2,
         std::acos(-nor(1)) / M_PI};
  return intersection<Real>{&s, r, t, hit_pos, nor, uv};
}
//------------------------------------------------------------------------------
template <floating_point Real, size_t N>
std::optional<intersection<Real>> check_intersection(
    sphere<Real, N> const& s, ray<Real, N> const& r, Real const min_t = 0) {
  return check_intersection(r, s, min_t);
}
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif
