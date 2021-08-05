#ifndef VC_ODEINT_SOLVER_HH
#define VC_ODEINT_SOLVER_HH
//=============================================================================
# include <cassert>
# include <cmath>
# include <cmath>
# include <type_traits>
# include <limits>
# include <tuple>
# include <vector>
//-----------------------------------------------------------------------------
# include "odeint_options.hh"
# include "odeint_evaldy.hh"
# include "odeint_stepperstate.hh"
# include "odeint_hermitespline.hh"
# include "odeint_generator.hh"
# include "odeint_helper.hh"
# include "odeint_output.hh"

# include "odeint_rk43.hh"
//=============================================================================
namespace VC {
namespace odeint {
//=============================================================================

/** Define problem dimensions and data types.
 */
template <int D,
          typename R, typename V,
          bool NormControl = false>
struct ode_t {

  static_assert(D>0, "invalid dimension D");
  static_assert(std::is_floating_point<R>::value,
                "expect floating point type R");

  using real_t = R;
  using vec_t = V;

  //---------------------------------------------------------------------------

  using helper = detail::helper_t<vec_t, real_t>;
  using output_t = detail::output_t<vec_t, real_t>;

  template <typename Sink>
  using sink_t = typename output_t::template sink_t<Sink>;

  template <typename Generator, typename Sink>
  using dense_output_t =
      typename output_t::template dense_output_t<Generator, Sink>;

  //---------------------------------------------------------------------------

  static constexpr auto dim() { return D; }
  static constexpr bool normcontrol() { return NormControl; }

  static auto norm(const vec_t& _v) {
    if constexpr (std::is_same<vec_t, real_t>::value)
      return std::abs(_v);
    else if constexpr (normcontrol())
      return vector_operations_t<vec_t>::norm2(_v);
    else
      return vector_operations_t<vec_t>::norminf(_v);
  }

  static auto abs(const vec_t& _v) {
    if constexpr (std::is_same<vec_t, real_t>::value)
      return std::abs(_v);
    else
      return vector_operations_t<vec_t>::abs(_v);
  }
  static auto max(const vec_t& _x, const vec_t& _y) {
    if constexpr (std::is_same<vec_t, real_t>::value)
      return std::max(_x, _y);
    else
      return vector_operations_t<vec_t>::max(_x, _y);
  }

  static auto qurt(real_t _x) { // x^(1/4)
    assert(_x >=0);
    return std::sqrt(std::sqrt(_x));
  }

  template <int P>
  static auto rt(real_t _x) { // x^(1/P)
    assert(_x >=0);
    static_assert(P >= 1, "expect positive P");
    if constexpr(P == 1) return _x;
    if constexpr(P == 2) return sqrt(_x);
    if constexpr(P == 3) return cbrt(_x);
    if constexpr(P == 4) return qurt(_x);
    else                 return pow(_x, real_t(1)/P);
  }

  auto static constexpr INF = std::numeric_limits<real_t>::infinity();
  auto static constexpr EPS = std::numeric_limits<real_t>::epsilon();
  auto static constexpr RHO = real_t(0.8);

  //---------------------------------------------------------------------------

  template <typename... Args>
  static auto make_options(Args&&... args) {
    return options_t { std::forward<Args>(args)... };
  }

  using options_t = options::odeopts_t<real_t>;

  using maybe_vec = maybe<vec_t>;
  using maybe_real = maybe<real_t>;

  //---------------------------------------------------------------------------

  using spline_t = hermite::spline_t<vec_t, real_t>;

  //---------------------------------------------------------------------------

  /** Define solver for a specific `Stepper` (e.g., rk43_t).
   */
  template <typename Stepper>
  struct solver_t : public Stepper {

    real_t t;
    real_t tdir;
    real_t tfinal;
    real_t tnew;
    real_t absh;
    real_t relerr;

    int nsteps = 0;
    evstate_t state = evstate_t::UNINITIALIZED;
    bool rejected = false;

    const options_t opts;

    using self_t = solver_t<Stepper>;
    using stepper_t = Stepper;
    stepper_t stepper;

    solver_t(const options_t& _opts = options_t{}) : opts(_opts) {}

    real_t h() const { return tdir * absh; }
    bool done() const { return this->t == this->tfinal; }

    real_t hmin() const { return (16*EPS)*std::abs(t); }

    /// Is `_t` in interval `[t, tnew]`?
    bool is_inside(real_t _t) const {
      if (tdir>0) return t <= _t && _t <= tnew;
      else        return tnew <= _t && _t <= t;
    }

    //-------------------------------------------------------------------------

    // TODO: obtain default interpolator from stepper -- type definition is enough(!)

    template <bool OnlyValue=false>
    auto hermite_interpolator() const {
      return hermite::interpolator_t<vec_t, real_t, OnlyValue> {
        this->y, this->dy, this->ynew, this->dynew, t, tnew
      };
    }
    template <typename Sink>
    auto hermite_interpolator_into(Sink&& _sink) const {
      constexpr bool OnlyValue =
          helper::template pass_value_only_t<Sink>::value;

      return helper::evaluate_function_into(
          hermite_interpolator<OnlyValue>(),
          std::forward<Sink>(_sink));
    }

    using solution_t = typename stepper_t::solution_t;

    //-------------------------------------------------------------------------

    static constexpr real_t AY = 1;
    static constexpr real_t ADY = 1;

    vec_t scale(const vec_t& _absy) const {
      return (AY * abs(_absy)) * opts.rtol + opts.atol;
    }
    vec_t scale(const vec_t& _absy, const vec_t& _absdy, real_t _absh = 1) const {
      return (AY * _absy + ADY * _absh * abs(_absdy)) * opts.rtol + opts.atol;
    }

    real_t scaled_norm(const vec_t& _y) const {
      return scaled_norm(_y, scale(abs(_y)));
    }
    real_t scaled_norm(const vec_t& _y, const vec_t _scale) const {
      if constexpr (normcontrol()) {
        return norm((_y/_scale)*(real_t(1)/dim())); // RMS
      }
      else {
        return norm(_y/_scale);
      }
    }

    static constexpr real_t RHO = real_t(0.8); // safety factor
    static constexpr real_t MAXTRIES = 100;    // for single step

    // TODO: provide also a simpler initialization (w/ only one evaluation at (t0,y0))

    /** Determine initial step size (evaluates `_dy` once).

        \pre Attributes `y, dy, t, tfinal,bi tdir` are initialized.

        Does one additional Euler step (one evaluation of `_dy`), see
        *Hairer. Solving Ordinary Differential Equations I. p. 169*.

        Code adapted from
        https://github.com/JuliaDiffEq/ODE.jl/blob/master/src/ODE.jl
     */
    template <typename DY>
    real_t guess_inital_step_size(DY&& _dy) {
      assert(state == evstate_t::OK);

      real_t normy = norm(this->y);
      real_t tau = std::max(opts.rtol * normy, opts.atol);
      // Note: This is not exactly scaled_norm() (Hairer: same norm).
      //       Should be good enough.

      real_t d0 = normy / tau;
      real_t d1 = norm(this->dy) / tau;

      real_t h0;

      if ((d0 < real_t(1e-5)) || (d1 < real_t(1e-5)))
        h0 = 1e-6;
      else
        h0 = (d0/d1)*real_t(0.01);

      // perform Euler step

      auto guard = require_no_undefined<vec_t> {};
      auto fdy = guard(helper::require_finite(_dy));

      vec_t y1 = this->y + tdir*h0*this->dy;
      vec_t dy1 = fdy(t + tdir*h0, y1);

      if (guard.undefined()) {
        return tdir * h0/2;             // leave to step size control
      }

      // estimate second derivative

      real_t d2 = norm(dy1 - this->dy)/(tau*h0);

      real_t h1;

      if (std::max(d1, d2) <= real_t(1e-15)) {
        h1 = std::max(real_t(1e-6), real_t(1e-3) * h0);
      }
      else {
        h1 = rt<stepper_t::order1()>(real_t(0.01)/std::max(d1, d2));
      }

      real_t hguess = tdir*std::min(std::min(100*h0, h1), tdir*(tfinal-t));

      return hguess;
    }

    real_t relative_error(const vec_t& _yerr) const {
      vec_t absy = max(abs(this->y), abs(this->ynew));

      return scaled_norm(_yerr, scale(absy, abs(this->dy), absh));
    }
    void set_relative_error(const vec_t& _yerr) {
      relerr = relative_error(_yerr);
    }

    bool is_acceptable() const {
      static constexpr real_t THRESHOLD = real_t(1.1);

      return relerr <= THRESHOLD;
    }

    real_t decreased_step_size() const {
      static constexpr real_t MINSCALE = 0.2;

      constexpr int P = stepper_t::order0();
      auto scale = RHO / rt<P>(relerr);          // TODO: check !!!
      assert(scale < 1);
      return absh * std::max(MINSCALE, scale);
    }

    real_t increased_step_size() const {
      static constexpr real_t MAXSCALE = 5;
      static constexpr real_t THRESHOLD_INCREASE = 0.5;

      if ((relerr < THRESHOLD_INCREASE) && (absh < opts.hmax)) {
        // Don't increase if absh was capped to hmax.
        constexpr int Q = stepper_t::order1();
        auto scale = RHO / rt<Q>(relerr);         // TODO: check !!!
        assert(scale/RHO > 1); // TODO: just cap?
        return std::min(opts.hmax,
                        absh * std::max(std::min(MAXSCALE, scale), real_t(1)));
      }
      return absh;
    }

    real_t bisected_step_size() const {
      static constexpr real_t BISECTION_FACTOR = 0.25;

      return absh * BISECTION_FACTOR;
    }


    /** Ensure that `tfinal` is is hit exactly.
        The factor `STRETCH_FINAL`
        - allows stretching the final step if `>1`, i.e., "overshoot";
          increases `absh`
        - requires a "safety margin" if `tfinal` is reached with `t+h`
          but not with `t+STETCH_FINAL*h`, i.e., don't take full step;
          decreases `hmax`
        \return new time `t+h()` or `tfinal`; stretching changes `absh`
     */
    real_t reach_destination() {
      static constexpr real_t STRETCH_FINAL = 1.1;

      assert(STRETCH_FINAL > 0);

      auto s = std::max(STRETCH_FINAL, real_t(1));

      if (s * absh > std::abs(tfinal - t)) {
        if (s >= 1) {
          return tfinal;
        }
        else if (STRETCH_FINAL * absh < std::abs(tfinal - t)) {
          return t + h() * std::min(STRETCH_FINAL, real_t(0.5));
        }
        else {
          return tfinal;
        }
      }
      else
        return t + h();
    }

    /** Initialize solver for subsequent call(s) to integrate() or step().
        \param _dy
        \param _t0 start time
        \param _t0 end time
        \param _y0 start point
     */
    template <typename DY>
    auto initialize(DY&& _dy, real_t _t0, real_t _t1, const vec_t& _y0) {
      state = evstate_t::OK;

      this->t = _t0;
      this->tfinal = _t1;
      this->tdir = (_t1 - _t0) >= 0 ? +1 : -1;
      this->y = _y0;
      this->nsteps = 0;
      this->rejected = false;

      auto guard = require_no_undefined<vec_t> {};
      auto fdy = guard(helper::require_finite(std::forward<DY>(_dy)));

      this->dy = fdy(_t0, _y0);

      if (guard.undefined()) {
        // VC_DBG_TRACE("initialization failed: could not evaluate dy(t0,y0)");
        return evstate_t::Failed;
      }

      this->ynew = this->y;
      this->dynew = this->dy;
      this->tnew = this->t;

      real_t hinit = opts.h0 > 0 ? opts.h0 : guess_inital_step_size(_dy);
      absh = std::abs(hinit);

      return evstate_t::OK;
    }

    template <typename DY>
    auto step(DY&& _dy) {
      assert(state == evstate_t::OK);

      tnew = this->reach_destination();
      absh = std::abs(tnew-t);

      real_t hmin = this->hmin();

      ++nsteps;

      int n = 0;

      // Note: The rejected attribute is used to
      //       1. prevent an increase after a step that was first rejected;
      //       2. prevent an increase during an "outer" bisection
      //          (due to rejection of a successful step by the output()
      //           callback).

      while (n < MAXTRIES) {
        auto result = this->_step(std::forward<DY>(_dy), t, h());

        ++n;

        if (result.state() == evstate_t::OK) {
          const auto yerr = result.value();

          set_relative_error(yerr);

          if (is_acceptable()) {

            // Don't increase step size if last step was rejected.
            if (!rejected)
              absh = increased_step_size();

            rejected = false;

            return result.state();
          }
          else {
            rejected = true;

            absh = std::max(decreased_step_size(), hmin);
            // TODO: Matlab ode54 uses absh/2 after first increase ("failure")
          }
        }
        else {
          assert((result.state() == evstate_t::Failed ||
                  result.state() == evstate_t::OutOfDomain)
                 && "dy(t,y) may only indicate Failed or OutOfDomain!");
          rejected = true;
          absh = bisected_step_size();
        }

        tnew = t + h();

        if (absh <= hmin) {
          if (result.state() != evstate_t::OK) {
            return result.state();
          }

          // No more progress in t!
          // VC_DBG_TRACE("integration tolerance not met");
          // VC_DBG_P(t);
          // VC_DBG_P(hmin);

          return evstate_t::OutOfTolerance;
        }
      } // while ntries

      // VC_DBG_TRACE("Exceeded MAXTRIES for integration step.");

      return evstate_t::Failed;
    }

    template <typename Output>
    constexpr evstate_t
    evaluate_output(Output&& _output,
                    [[maybe_unused]] evstate_t _err = evstate_t::Failed) const {
      static_assert(std::is_invocable<Output,decltype(*this)>::value,
                    "Output must be invocable with stepper as argument!");

      using tp = decltype(_output(*this));

      if constexpr(std::is_same<tp, void>::value) {
        _output(*this);
        return evstate_t::OK;
      }
      else if constexpr(std::is_same<tp, bool>::value) {
        return _output(*this) ? evstate_t::OK : _err;
      }
      else if constexpr(std::is_same<tp, evstate_t>::value) {
        return _output(*this);
      }
      else {
        static_assert(!std::is_same<tp, evstate_t>::value,
                      "output() must return void, bool or evstate_t");
        return evstate_t::Failed;
      }
    }

    static constexpr int MAXSTEPS = 0x10000;

    template <typename DY, typename Output>
    auto _integrate(DY&& _dy, Output&& _output) {
      if (state != evstate_t::OK) {
        return state;
      }

      int maxsteps = opts.maxsteps > 0 ? opts.maxsteps : MAXSTEPS;

      auto ostate = evaluate_output(_output);

      assert(ostate == evstate_t::OK &&
             "output() must succeed for initial point (t0,y0)");

      while (!this->done() && nsteps < maxsteps) {

        auto newstate = this->step(_dy);

        if (newstate == evstate_t::OK ||
            newstate == evstate_t::OutOfDomain) {

          ostate = evaluate_output(_output);

          if (ostate != evstate_t::OK) {
            if (ostate == evstate_t::Stopped) {
              newstate = evstate_t::Stopped;
            }
            else if (ostate == evstate_t::OutOfDomain) {

              if ((absh=bisected_step_size()) > hmin()) {
                --nsteps;
                rejected = true;
                continue;                       // repeat step
              }

              newstate = ostate;
            }
            else {
              // VC_DBG_P(ostate);
              assert(!"invalid state: output() may indicate Stop or OutOfDomain!");
              return state = evstate_t::Failed;
            }
          }

          this->t= this->tnew;
          this->y= this->ynew;
          this->dy= this->dynew;
        }
        if (newstate != evstate_t::OK) {
          return state = newstate;
        }
      } // while !done

      // Note: special case for output if done !?!

      if (nsteps > maxsteps) {
        // VC_DBG_TRACE("exceeded maximum number of steps");
        return evstate_t::Stopped;
      }

      return state;
    }

    /// Integrate `_dy`. Requires initialize() called immediately before.
    template <typename DY>
    auto integrate(DY&& _dy) {
      return integrate(std::forward<DY>(_dy), [](const auto&){});
    }
    /// Integrate `_dy`. Requires initialize() called immediately before.
    template <typename DY, typename Output>
    auto integrate(DY&& _dy, Output&& _output) {
      return _integrate(std::forward<DY>(_dy), std::forward<Output>(_output));
    }

    /** Initialize solver and integrate `_dy` (calls initialize()).
        \param _dy computes derivative as `_dy(t,y)`
        \param _t0 start time
        \param _t1 end type
        \param _y0 initial position
        \param _output optional output function
     */
    template <typename DY, typename Output>
    auto integrate(DY&& _dy, real_t _t0, real_t _t1, const vec_t& _y0,
                   Output&& _output) {
      auto result = initialize(std::forward<DY>(_dy), _t0, _t1, _y0);
      if (result != evstate_t::OK)
        return result;

      return _integrate(std::forward<DY>(_dy), std::forward<Output>(_output));
    }
    /// Initialize solver and integrate `_dy` (calls initialize()).
    template <typename DY>
    auto integrate(DY&& _dy, real_t _t0, real_t _t1, const vec_t& _y0) {
      return  integrate(std::forward<DY>(_dy), _t0, _t1, _y0,
                        [](const auto&){});
    }
  }; // stepper_base_t

  /** Output

      ## Example:

      If `ode` defines an ODE solver type and `solution` and
      `dense_solution` are splines of type `ode::spline_t`, the
      following defines an "output pipeline"

      ```
      Output
        >>
          // Reject if step passed y[0] == 0.
          ode::predicate([](const auto& stepper) {
            return stepper.ynew[0] >=0;
          }, OutOfDomain)
         >>
          // Generate 5 uniformly spaced samples and output to `cerr`.
          ode::dense([](auto t, const auto& y, const auto& dy) {
                        std::cerr << t << '\t' << y << std::endl;
                     },
                     ode::generate(5))
         >>
           // Pass result of every step and otuput to `cout`.
           ode::sink(
                 [](auto t, const auto& y) {
                   std::cout << "SINK: " << t << '\t' << y << std::endl;
                 })
         >>
           // Pass result of every step and store in `solution`
           ode::sink(solution)
         >>
           // Generate 15 uniformly spaced samples and store in dense_solution.
           ode::dense(dense_solution, ode::generate(15))
      ```

      @{
   */

  /// Generate dense() output at uniform steps of size `_delta`.
  static auto generate(real_t _delta) {
    using generator::generator_t, generator::tag_t;
    return generator_t<tag_t::uniform_delta_steps, real_t>(_delta);
  }

  /// Generate dense() output at `_n` steps with uniform step size.
  static auto generate(int _n) {
    using generator::generator_t, generator::tag_t;
    return generator_t<tag_t::n_uniform_steps, real_t>(_n);
  }

  /// Generate dense() output for times in range.
  template <typename Iterator>
  static auto generate(Iterator _begin, Iterator _end) {
    using generator::generator_t, generator::tag_t;
    return generator_t<tag_t::time_range, real_t, Iterator> {
             _begin, _end
           };
  }

  /// Generate dense output at times `_f(i)` for `i=0,1,...,_n-1`.
  template <typename F>
  static auto generate(F&& _f, int _n){
    using generator::generator_t, generator::tag_t;
    return generator_t<tag_t::indexed_time_range, real_t, F> {
      std::forward<F>(_f), _n
    };
  }

  // Note: Dense output does not stop integration after last output!

  /// Generate dense output and store as spline.
  template <typename Generator>
  static auto dense(spline_t& _solution, Generator&& _generator) {
    return dense([&](auto t, auto y, auto dy) {
                               _solution.push_back(t, y, dy);
                             }, std::forward<Generator>(_generator));
  }

  /// Generate dense output into sink().
  template <typename Generator, typename Sink>
  static auto dense(Sink&& _sink, Generator&& _generator) {
    return dense_output_t<Generator, Sink>(
        std::forward<Generator>(_generator),
        std::forward<Sink>(_sink));
  }

  /// A sink accepts output as `sink(t,y)` or `sin(t,y,dy)`.
  template <typename Sink>
  static auto sink(Sink&& _sink) {
    return sink_t<Sink> { std::forward<Sink>(_sink) };
  }

  /// Make sink that stores output in Hermite spline.
  static auto sink(spline_t& _spline) {
    return sink([&](real_t t, const vec_t& y, const vec_t& dy) {
                       _spline.push_back(
                           typename spline_t::value_type { t, y, dy });
                     });
  }

  // TODO: Distinguish between Hermite spline and "stepper native" representation.


  /** Turn `_predicate(const Stepper&) -> bool` into output function.

      Yields lambda with transformed return values: `true` is
      `AcceptStep`, `false` is `_err`.

      \ sa VC::ode::in_domain
   */
  template <typename Predicate>
  static auto
  predicate(Predicate&& _predicate, evstate_t _err = evstate_t::Failed) {
    auto predicate = std::forward<Predicate>(_predicate);
    return [_err, predicate](const auto& _stepper) {
             return _stepper.template evaluate_output(predicate, _err);
           };
  }

  /** Is step in domain?

      Output function that is used as a predicate on the current,
      tentative step. Checks if step is inside domain, and returns
      `OutOfDomain` if not.

      Note that this predicate is evaluated only if both `(s.t,s.y)`
      and `(s.tnew,s.ynew)` of stepper `s` are *in* the domain. This
      may, however, be not sufficient for testing the step as a whole:
      the trajectory of the step may leave the domain. (Consider,
      e.g., a non-convex domain.) The purpose of this predicate is to
      test the "whole step".

      \sa VC::ode::predicate
   */
  template <typename Predicate>
  static auto
  in_domain(Predicate&& _predicate) {
    auto pred = std::forward<Predicate>(_predicate);
    return predicate([pred](auto&& _stepper) {
                       return pred(std::forward<decltype(_stepper)>(_stepper));
                     }, evstate_t::OutOfDomain);
  }

  /// same as VC::odeint::output()
  template <typename... Args>
  static auto output(Args&&... args) {
    return detail::output(std::forward<Args>(args)...);
  }

  /// same as VC::odeint::Output
  static constexpr detail::empty_output_pipeline_t Output {};

  /// @}

  /** Define steppers and solvers.
     @{
   */

  using rk43_t = steppers::rk43_t<vec_t, real_t>;

  auto static solver(rk43_tag, const options_t& _opts = options_t{}) {
    return solver_t<rk43_t>(_opts);
  }

  /// @}

}; // struct ode_t

//=============================================================================
}  // namespace odeint
}  // namespace VC
//=============================================================================
#endif  // VC_ODEINT_SOLVER_HH
