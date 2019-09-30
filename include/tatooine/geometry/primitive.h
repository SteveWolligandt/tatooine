#ifndef TATOOINE_GEOMETRY_PRIMITIVE_H
#define TATOOINE_GEOMETRY_PRIMITIVE_H

#include "../tensor.h"

//==============================================================================
namespace tatooine::geometry {
//==============================================================================

template <typename Real, size_t N>
struct primitive {
  static constexpr auto num_dimensions() { return N; }
  using real_t = Real;
  using pos_t  = vec<Real, N>;

  virtual bool is_inside(const pos_t& x) const = 0;
};

//==============================================================================
}  // namespace tatooine::geometry
//==============================================================================

#endif
