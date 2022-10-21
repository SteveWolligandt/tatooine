#ifndef TATOOINE_ANALYTICAL_NUMERICAL_H
#define TATOOINE_ANALYTICAL_NUMERICAL_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/math.h>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
/// From here:
/// http://web.cse.ohio-state.edu/~crawfis.3/Data/Tornado/tornadoSrc.c
template <typename Real>
struct tornado : vectorfield<tornado<Real>, Real, 3> {
  using this_type   = tornado<Real>;
  using parent_type = vectorfield<this_type, Real, 3>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  //----------------------------------------------------------------------------
  static constexpr real_type eps = 1e-10;
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& pos, real_type const t) const
      -> tensor_type {
    // For each z-slice, determine the spiral circle.
    //    (xc,yc) determine the center of the circle.
    auto const c = vec{0.5 + 0.1 * sin(0.04 * t + 10 * pos.z()),
                       0.5 + 0.1 * cos(0.03 * t + 3 * pos.z())};

    // The radius also changes at each z-slice.
    auto r = 0.1 + 0.4 * pos.z() * pos.z() + 0.1 * pos.z() * sin(8 * pos.z());
    // r is the center radius, r2 is for damping
    auto const r2    = 0.2 + 0.1 * pos.z();
    auto temp  = euclidean_distance(pos.xy(), c);
    auto       scale = abs(r - temp);
    //  I do not like this next line. It produces a discontinuity in the
    //  magnitude. Fix it later.
    if (scale > r2) {
      scale = 0.8 - scale;
    } else {
      scale = 1;
    }

    auto z0 = 0.1 * (0.1 - temp * pos.z());
    if (z0 < 0) {
      z0 = 0;
    }
    temp  = sqrt(temp * temp + z0 * z0);
    scale = (r + r2 - temp) * scale / (temp + eps);
    scale = scale / (1 + pos.z());
    return tensor_type{(pos.y() - c.y()) + 0.1 * (pos.x() - c.x()),
                       -(pos.x() - c.x()) + 0.1 * (pos.y() - c.y()), z0} *
           scale;
  }
};
//==============================================================================
tornado()->tornado<double>;
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
