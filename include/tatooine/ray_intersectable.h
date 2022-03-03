#ifndef TATOOINE_RAY_INTERSECTABLE_H
#define TATOOINE_RAY_INTERSECTABLE_H
//==============================================================================
#include <tatooine/intersection.h>
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, std::size_t NumDimensions>
struct ray_intersectable {
  using real_type = Real;
  static_assert(is_floating_point<real_type>);
  using intersection_type          = intersection<real_type, NumDimensions>;
  using optional_intersection_type = std::optional<intersection_type>;
  using ray_type                   = ray<real_type, NumDimensions>;
  //============================================================================
  virtual auto check_intersection(ray_type const& r,
                                  real_type const min_t) const
      -> optional_intersection_type = 0;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
