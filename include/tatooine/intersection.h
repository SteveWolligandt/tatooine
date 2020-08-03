#ifndef TATOOINE_GEOMETRY_INTERSECTION_H
#define TATOOINE_GEOMETRY_INTERSECTION_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number Real, size_t N>
struct ray_intersectable;
//==============================================================================
/// Intersections are created when a \ref cg::Ray "ray" hits an \ref
/// cg::primitive "primitive object".
template <real_number Real, size_t N>
struct intersection {
  using real_t = Real;
  static constexpr size_t num_dimensions() {
    return N;
  }
  /// pointer to primitive object
  const ray_intersectable<Real, N>* intersectable;
  /// incident ray
  ray<Real, N> incident_ray;
  /// position on ray
  Real t;
  /// world position of intersection
  vec<Real, N> position;
  /// normal of intersection point on the primitive
  vec<Real, N> normal;
  /// uv-coordinate on the primitive
  vec<Real, N - 1> uv;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
