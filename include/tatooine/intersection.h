#ifndef TATOOINE_GEOMETRY_INTERSECTION_H
#define TATOOINE_GEOMETRY_INTERSECTION_H
//==============================================================================
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N> struct ray_intersectable;
//==============================================================================
/// Intersections are created when a \ref cg::Ray "ray" hits an \ref
/// cg::primitive "primitive object".
template <typename Real, size_t N>
struct intersection {
  /// pointer to primitive object
  const ray_intersectable<Real, N>* intersectable;
  /// incident ray
  ray<Real, N> incident_ray;
  /// position on ray
  double t;
  /// world position of intersection
  vec<Real,N> position;
  /// normal of intersection point on the primitive
  vec<Real, N> normal;
  /// uv-coordinate on the primitive
  vec<Real, N-1> uv;
};
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif
