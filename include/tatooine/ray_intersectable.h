#ifndef TATOOINE_RAY_INTERSECTABLE_H
#define TATOOINE_RAY_INTERSECTABLE_H
//==============================================================================
#include <tatooine/intersection.h>
#include <tatooine/ray.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_number Real, size_t N>
struct ray_intersectable{
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;
  using ray_t = ray<Real, N>;
  using intersection_t = intersection<Real, N>;
  //============================================================================
 public:
  virtual std::optional<intersection<Real, N>> check_intersection(
      const ray<Real, N>& r, const Real min_t = 0) const = 0;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
