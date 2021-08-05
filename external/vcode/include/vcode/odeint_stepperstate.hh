#ifndef VC_ODEINT_STEPPERBASE_HH
#define VC_ODEINT_STEPPERBASE_HH
//=============================================================================
namespace VC {
namespace odeint {
namespace steppers {
//=============================================================================

/** All steppers share this state.

    After each step, `y` and `dy` are the position and derivative at
    the step's origin; `ynew` and `dynew` are the position and
    derivative at the step's destination.

    `y` and `dy` are always defined (unless the initialization of the
    stepper failed). `ynew` and `dynew` may be undefined and/or may be
    rejected as the step may or may not be accepted.
 */
template <typename T>
struct stepper_base_state_t {
  using vec_t = T;

  vec_t y;
  vec_t dy;

  vec_t ynew;
  vec_t dynew;
};

//-----------------------------------------------------------------------------

/** Attributes used by particular steppers (e.g., rk43_stepper_t).

    The particular stepper inherits from this class, i.e., can
    access attributes, and it's functionality is mixed into
    `stepper_base_t<Stepper>` (which can access attributes and
    methods of the particular `Stepper`).

    \tparam N number of partial steps that are saved in `k`. The
    array `k` is used for interpolation/dense stepping for
         methods of order higher than 3. The special case `N==0`
         avoids extra storage and uses cubic Hermite interpolation
         as "fallback".
*/
template <typename T, size_t N>
struct stepper_state_t : public stepper_base_state_t<T> {
  using vec_t = T;

  std::array<vec_t, N> k;

  template <size_t K>
  void commit(const std::array<vec_t, K>& _k) {
    if constexpr (K == N)
      this->k = _k;
    else {
      static_assert(N == 0, "invalid size: store either all or no steps k");
      }
  }
};

//=============================================================================
}  // namespace steppers
}  // namespace odeint
}  // namespace VC
//=============================================================================
#endif  // VC_ODEINT_STEPPERBASE_HH
