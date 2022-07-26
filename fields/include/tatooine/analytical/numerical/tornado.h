#ifndef TATOOINE_ANALYTICAL_NUMERICAL_H
#define TATOOINE_ANALYTICAL_NUMERICAL_H
//==============================================================================
#include <tatooine/field.h>

#include <cmath>
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
      -> tensor_type final {
    real_type r, xc, yc, scale, temp, z0;
    real_type r2 = 8;

    // For each z-slice, determine the spiral circle.
    xc = 0.5 + 0.1 * std::sin(0.04 * t + 10 * pos.z());
    //    (xc,yc) determine the center of the circle.
    yc = 0.5 + 0.1 * std::cos(0.03 * t + 3 * pos.z());
    //  The radius also changes at each z-slice.
    r = 0.1 + 0.4 * pos.z() * pos.z() + 0.1 * pos.z() * std::sin(8 * pos.z());
    //    r is the center radius, r2 is for damping
    r2    = 0.2 + 0.1 * pos.z();
    temp  = std::sqrt((pos.y() - yc) * (pos.y() - yc) + (pos.x() - xc) * (pos.x() - xc));
    scale = std::fabs(r - temp);
    //  I do not like this next line. It produces a discontinuity in the
    //  magnitude. Fix it later.
    if (scale > r2) {
      scale = 0.8 - scale;
    } else {
      scale = 1;
    }
    z0 = 0.1 * (0.1 - temp * pos.z());
    if (z0 < 0) {
      z0 = 0;
    }
    temp       = std::sqrt(temp * temp + z0 * z0);
    scale      = (r + r2 - temp) * scale / (temp + eps);
    scale      = scale / (1 + pos.z());
    return tensor_type{ (pos.y() - yc) + 0.1 * (pos.x() - xc),
                    -(pos.x() - xc) + 0.1 * (pos.y() - yc),
                     z0} *
           scale;
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_type const& /*pos*/, real_type const /*t*/) const
      -> bool final {
    return true;
  }
};
tornado()->tornado<double>;
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
