#ifndef TATOOINE_INTERSECTION_H
#define TATOOINE_INTERSECTION_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct ray_intersectable;
//==============================================================================
template <typename Real, size_t N>
struct intersection {
  static_assert(is_floating_point<Real>);
  using real_type                  = Real;
  using vec_t                   = vec<Real, N>;
  using pos_type                   = vec_t;
  using normal_t                = vec_t;
  using ray_t                   = ray<real_type, N>;
  using ray_intersectable_ptr_t = ray_intersectable<real_type, N>;
  //============================================================================
  /// pointer to primitive object
  ray_intersectable_ptr_t const* intersectable;
  /// incident ray
  ray_t incident_ray;
  /// position on ray
  real_type t;
  /// world position of intersection
  pos_type position;
  /// normal of intersection point on the primitive
  normal_t normal;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
