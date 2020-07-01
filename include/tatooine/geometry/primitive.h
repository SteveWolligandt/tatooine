#ifndef TATOOINE_GEOMETRY_PRIMITIVE_H
#define TATOOINE_GEOMETRY_PRIMITIVE_H
//==============================================================================
#include <tatooine/tensor.h>
#include <tatooine/ray_intersectable.h>
//==============================================================================
namespace tatooine::geometry {
//==============================================================================
template <typename Real, size_t N>
struct primitive : ray_intersectable<Real, N> {
  static constexpr auto num_dimensions() { return N; }
  using real_t = Real;
  using pos_t  = vec<Real, N>;
  //----------------------------------------------------------------------------
  virtual ~primitive() = default;
};
//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================
#endif
