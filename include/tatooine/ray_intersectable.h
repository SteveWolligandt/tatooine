#ifndef TATOOINE_RAY_INTERSECTABLE_H
#define TATOOINE_RAY_INTERSECTABLE_H
//==============================================================================
#include <tatooine/intersection.h>
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct ray_intersectable {
  using real_t                  = Real;
  static_assert(is_floating_point<real_t>);
  using intersection_t          = intersection<real_t, N>;
  using optional_intersection_t = std::optional<intersection_t>;

  using ray_t = ray<real_t, N>;
  //============================================================================
  virtual auto check_intersection(ray_t const& r, real_t const min_t) const
      -> optional_intersection_t = 0;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
