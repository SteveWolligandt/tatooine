#ifndef VC_ODEINT_EVALDY_HH
#define VC_ODEINT_EVALDY_HH
//=============================================================================
# include <cassert>
# include <utility>     // std::forward
# include <type_traits> // std::is_*
# include <iostream>    // define operator>>
//=============================================================================
namespace VC {
namespace odeint {
//=============================================================================

/** Evaluation states for `dy(t,y)` and the integrator.

    We collect all states in this `enum`. Some of them are for
    "public" use, e.g., can be indicated on evaluation of `dy(t,y)`
    and/or communicated to the user. Others are for "private" use
    "within" the integrator.

    The rule is that "user code" should not user the `enum` fields but
    public constants.
 */
enum class evstate_t {
  OK,
  Failed,
  OutOfDomain,
  Stopped,
  OutOfTolerance,
  UNINITIALIZED
};

//-----------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& _out, evstate_t _state) {
  const char* p = nullptr;
  switch (_state) {
    case evstate_t::OK:
      p = "success";
      break;
    case evstate_t::Failed:
      p = "failed";
      break;
    case evstate_t::OutOfDomain:
      p = "out-of-domain";
      break;
    case evstate_t::Stopped:
      p = "stopped";
      break;
    case evstate_t::OutOfTolerance:
      p = "out-of-tolerance";
      break;
    case evstate_t::UNINITIALIZED:
      p = "(uninitialized)";
      assert(!"State must be initialized!");
      break;
    default:
      assert(!"invalid state");
  }
  return _out << p;
}

//-----------------------------------------------------------------------------

// NOTE:
//
// We could use std::variant<> but I prefer the tuple<> to the union
// (although one element may be undefined), and I want construction
// with a "success state".
//

//
// NOTE: undefined_result_of, maybe_t, require_no_undefined_t are general
//       helpers; they should be parametrized in an ode *namespace*
//       with State=evaluate_state_t; thy can be defined precisely
//       in the ode_solver<VecType> class.
//
//       Use can defined dy as -> maybe<VecType> and does not need to care
//       otherwise!
//

/** "Define" construction of an undefined result.

    As we don't use a `union`, we always need to construct a result,
    which may be undefined (i.e., not initialized).

    For debugging it makes sense to initialize with "undefined" values
    if possible like `NaN` for floating point types.
 */
template <typename Result>
struct undefined_result_of {
  auto operator()() const { return Result { }; }
};

/** Traits on `evstate_t` for generalized states in maybe_t.
    \sa maybe_t
 */
template <typename State>
struct state_traits_t {
  state_traits_t() {
    static_assert(!std::is_same<State,State>::value,
                  "missing specialization of state_traits_t or "
                  "not a state type");
  }
};

template <> struct state_traits_t<evstate_t> {
  static constexpr auto success() { return evstate_t::OK; }
};


/** `maybe_t(T)` holds immutable result value and state.

    `maybe_t` is comparable to`std::variant<Result, State>` but

    - does not use a `union` (instead there exists always a value,
      which might be constructed by undefined_result_of()).
    - is much simpler as it uses `State` to discriminate undefined
      values and does not `throw`.

    Note that

    - Any "defined" `Result` can be promoted to a well-defined
      `maybe_t<Result>` by a constructor.
    - There is a helper maybe() that construct
      `maybe_t<T,evstate_t>`.
 */
template <typename Result, typename State>
class maybe_t {
 public:
  using state_t = State;
  using value_t = Result;

  static constexpr auto SUCCESS = state_traits_t<State>::success();

  maybe_t(const value_t& _value, state_t _state) :
      m_value(_value), m_state(_state) {}
  maybe_t(const value_t& _value) : m_value(_value), m_state(SUCCESS) {}
  maybe_t(state_t _state) : maybe_t(undefined_result_of<value_t>{}(), _state) {
    assert(_state != SUCCESS && "missing value: must provide value on success");
  }
  maybe_t(const maybe_t& _other) : maybe_t(_other.m_value, _other.m_state) {}

  auto state() const { return m_state; }
  auto value() const {
    assert(is_defined() && "undefined value");
    return m_value;
  }
  auto maybe_value() const { return m_value; }
  bool is_defined() const {
    return m_state == state_traits_t<State>::success();
  }
 private:
  const value_t m_value;
  const state_t m_state;
};

/** Do multiple evaluations, where none of them may fail.

    For numerical integration, the function `dy(t,y)` is evaluated
    multiple times in the stepper. All evaluations required for a
    single step must yield a valid result. However, we assume that
    `dy` in fact returns `maybe_t`: it may fail, e.g., because `(t,y)`
    is out of domain.

    This helper

    - Ensures that `dy` is treated as if it returned `maybe_t<Result,
      State>`, regardless whether it returns `Result` or `maybe_t`.
    - Tracks state to ensures that no further evaluation is computed
      after `dy` failed. The state can be queried by the state() and
      no_undefined() methods.

    The instance tracks state, and `operator()(F dy)` "rewrites" `dy`
    such that the above statements hold: it returns a new lambda.

    Note that

    - we assume that `F` returns `Result` or `maybe_t<Result, State>`
      (or any type that is promoted). We make no assumptions on the
      arguments of `F`.
    - There is a helper require_no_undefined<T> that construct
      `require_no_undefined<T,evstate_t>`.
 */
template <typename Result, typename State>
class require_no_undefined_t {
 public:
  using state_t = State;
  using value_t = Result;

  static constexpr auto SUCCESS = state_traits_t<State>::success();

  state_t state() const { return m_state; }
  bool no_undefined() const { return m_state == SUCCESS; }
  bool undefined() const { return !no_undefined(); }

  template <typename F>
  auto operator()(F&& f) const {
    state_t& thestate = m_state;
    return [f,&thestate](auto&&... args) {
             if (thestate == SUCCESS) {
               auto result = maybe_t<Result, State> {
                 f(std::forward<decltype(args)>(args)...) };
               thestate = result.state();
               return result.maybe_value();
             }
             else {
               return undefined_result_of<value_t>{}();
             }
           };
  }

 private:
  mutable state_t m_state = SUCCESS;
};

//-----------------------------------------------------------------------------

template <typename V>
using maybe = maybe_t<V,evstate_t>;

template <typename V>
using require_no_undefined = require_no_undefined_t<V,evstate_t>;

//-----------------------------------------------------------------------------

/** \defgroup odeint_evaldy Evaluation state of `dy(t,y)`.

    The following constants should be used to communicate state after
    trying to evaluate the user defined `dy(t,y)`.

    Note that evstate_t may define more states, they are reserved for
    internal use.

    ## Evaluation of `dy(t,y)`

    If `ode` defines an ODE solver type, the evaluation of `dy(t,y)`
    should return `ode::maybe_vec` (see VC::odeint::maybe_t), i.e., it
    returns either a vector or `Failed` or `OutOfDomain`. In the first
    case the returned vector must be of finite size (i.e., the user is
    responsible for testing for undefined values).

    If `dy(t,y)` returns a plain vector, the result is converted to
    `ode::maybe_vec` such that any result that is not finite will be
    interpreted as `Failed`.

    In both cases (`Failed` and `OutOfDomain`), the solver will reduce
    the step size such that the integration either can continue or
    that determine or stops at the "last" domain point where `dy`
    could be evaluated. The solver will return the failure state, for
    `OutOfDomain` the position will be an approximate boundary point.

    Example for `dy` with domain restricted to the half-space
    `y[0]>=0`:

    ```
    auto dy(double t, const vec2& y) -> ode::maybe_vec {
       if (y[0] < 0)
         return OutOfDomain;
       else
         return vec2 { -y[1], y[0] };
    };
    ```

    ## Evaluation of `output(const Stepper&)`

    An output function may return types

    - `void`: always `AcceptStep`
    - `bool`: `true` is `AccepStep`, `false` is `Fail`
      (see also \ref odeint_output)
    - `AcceptStep`, `OutOfDomain`, or `Stop`

    Note that an output function *cannot* return `Fail`.

    `Stop` will terminte the integration. `OutOfDomain` will start a
    bisection as for `dy(t,y)`. `OutOfDomain` on output is provided to
    enable precise testing for non-convex boundaries or "think
    barriers" which otherwise may be passed with a large enough step
    size.

    @{
 */

inline const auto Failed = evstate_t::Failed;
inline const auto OutOfDomain = evstate_t::OutOfDomain;
inline const auto Stop = evstate_t::Stopped;
inline const auto AcceptStep = evstate_t::OK;

/// @}

//=============================================================================
} // namespace odeint
} // namespace VC
//=============================================================================
#endif // VC_ODEINT_EVALDY_HH
