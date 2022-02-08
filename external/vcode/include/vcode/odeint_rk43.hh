#ifndef VC_ODEINT_RK43_HH
#define VC_ODEINT_RK43_HH
//=============================================================================
# include <cassert>
# include <cmath>
//-----------------------------------------------------------------------------
# include "odeint_evaldy.hh"
# include "odeint_stepperstate.hh"
# include "odeint_hermitespline.hh"
# include "odeint_helper.hh"
//=============================================================================
namespace VC {
namespace odeint {
//=============================================================================

// TODO: doc
const static struct rk43_tag {} RK43;

//-----------------------------------------------------------------------------
namespace steppers {
//-----------------------------------------------------------------------------

template <typename T, typename R>
struct rk43_t : public stepper_state_t<T, 0> {
  using real_type = R;
  using vec_t = T;

  using maybe_vec = maybe<vec_t>;
  using solution_t= hermite::spline_t<real_type, vec_t>;
  using helper = detail::helper_t<vec_t, real_type>;

  static constexpr int order0() { return 4; }
  static constexpr int order1() {
    return 4;
  }  // TODO: Is this correct for this tableau??

  template <typename DY>
  maybe_vec _step(DY&& _dy, real_type _t, real_type _h) {
    auto guard = require_no_undefined<vec_t>{};
    auto dy = guard(helper::require_finite(std::forward<DY>(_dy)));
    // dy: no evaluations of dy after any failure

    const real_type t= _t;
    const real_type h= _h;
    const vec_t y= this->y;

    std::array<vec_t, 4> k;

    k[0]= this->dy * h;  // = dy(t, _y)*h;
    k[1]= dy(t + h / 2, y + k[0] / 2) * h;
    k[2]= dy(t + h / 2, y + k[1] / 2) * h;
    k[3]= dy(t + h, y + k[2]) * h;

    constexpr real_type I6= real_type(1) / 6;
    constexpr real_type I3= real_type(1) / 3;

    if (guard.undefined())  // no change of attributes
      return guard.state();

    this->ynew= y + (k[0] * I6 + k[1] * I3 + k[2] * I3 + k[3] * I6);
    this->dynew= dy(t + h, this->ynew);

    if (guard.undefined())  // no change of attributes
      return guard.state();

    auto yerr= (k[3] - this->dynew * h) * I6;

    this->commit(k);

    return yerr;
  }
};

//=============================================================================
}  // namespace steppers
}  // namespace odeint
}  // namespace VC
//=============================================================================
#endif  // VC_ODEINT_RK43_HH
