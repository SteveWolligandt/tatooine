#ifndef TATOOINE_INTERSECTION_H
#define TATOOINE_INTERSECTION_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real>
struct ray_intersectable;
//==============================================================================
template <typename Real>
struct intersection {
  static_assert(is_floating_point<Real>);
  using real_t                  = Real;
  using vec3_t                  = vec<real_t, 3>;
  using vec2_t                  = vec<real_t, 2>;
  using pos_t                   = vec3_t;
  using uv_t                    = vec2_t;
  using ray_t                   = ray<real_t, 3>;
  using ray_intersectable_ptr_t = ray_intersectable<real_t>;
  //============================================================================
  /// pointer to primitive object
  ray_intersectable_ptr_t const* intersectable;
  /// incident ray
  ray_t incident_ray;
  /// position on ray
  real_t t;
  /// world position of intersection
  vec3_t position;
  /// normal of intersection point on the primitive
  vec3_t normal;
  /// uv-coordinate on the primitive
  vec2_t uv;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
