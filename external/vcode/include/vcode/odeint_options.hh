#ifndef VC_ODEINT_OPTIONS_HH
#define VC_ODEINT_OPTIONS_HH
//=============================================================================
# include <cassert>
# include <utility>     // std::forward
# include <type_traits> // std::is_*
# include <iostream>    // define operator>>
//=============================================================================
namespace VC {
namespace odeint {
//=============================================================================
namespace options {
//-----------------------------------------------------------------------------

/// Set option, for internal use with VC::odeint::odeopts_t
template <typename klass,typename T> struct option_t {
  constexpr option_t(const T& _value) : value(_value) {}
  using option_class = klass;
  const T value;
};

#define MAKE_OPTION_CLASS(name)                                            \
  struct name##_class {                                                    \
    template <typename T>                                                  \
    constexpr auto operator=(T value) const {                           \
      static_assert(std::is_arithmetic<T>::value, /* nicer error message*/ \
                    "option '" #name "' must be a scalar value");       \
      return option_t<name##_class, T>{value};                          \
    }                                                                   \
  }

MAKE_OPTION_CLASS(rtol);
MAKE_OPTION_CLASS(atol);
MAKE_OPTION_CLASS(h0);
MAKE_OPTION_CLASS(hmax);
MAKE_OPTION_CLASS(maxnsteps);

// TODO: Avoid the macro? (I'd like to keep a string like #name, though!)

# undef MAKE_OPTION_CLASS

//-----------------------------------------------------------------------------

/** Define options to ODE solvers.

 */
template <typename T>
class odeopts_t {
 public:
  using real_type  = T;

  template <typename... Args>
  constexpr odeopts_t(Args&&... args) { set(std::forward<Args>(args)...); }
  template <typename... Args>
  constexpr auto& set(Args&&... args) {
    _setlist(std::forward<Args>(args)...);
    check_assertions();
    return *this;
  }

  real_type rtol = real_type(1e-3); //!< relative tolerance
  real_type atol = real_type(1e-8); //!< absolute tolerance
  real_type h0 = 0;              //!< initial step size (none if 0)
  /// maximum step size
  real_type hmax = std::numeric_limits<real_type>::infinity();
  int    maxsteps = 0;        //!< maximum number of steps (if >0)

 private:
  template <typename klass> using opt_t = option_t<klass, real_type>;

  constexpr void check_assertions() const {
    assert(rtol > 0 && "relative tolerance must be positive");
    assert(atol > 0 && "absolute tolerance must be positive");
    assert(h0 >= 0 && "initial step size h0 must be nonnegative");
    assert(hmax > 0 && "maximum step size h0 must be positive");
    assert(maxsteps >= 0 && "maximum number of steps must be positive");
  }

  constexpr void _set(opt_t<rtol_class> opt) { rtol = opt.value; }
  constexpr void _set(opt_t<atol_class> opt) { atol = opt.value; }
  constexpr void _set(opt_t<h0_class> opt) { h0 = opt.value; }
  constexpr void _set(opt_t<hmax_class> opt) { hmax = opt.value; }
  // void _set(opt_t<maxnsteps_class> opt) { maxsteps = opt.value; }

  template <typename S>
  constexpr void _set(option_t<maxnsteps_class, S> opt) {
    static_assert(std::is_integral<S>::value,
                  "maximum number of steps must be an integer");
    maxsteps = opt.value;
  }
  // cast other types to real_type
  template <typename klass, typename R>
  constexpr void _set(option_t<klass, R> opt) {
    _set(opt_t<klass> { static_cast<real_type>(opt.value) });
  }
  // handle invalid option class (w/ real_type)
  template <typename InvalidOption>
  constexpr void _set(opt_t<InvalidOption>) {
    static_assert(!std::is_same<InvalidOption, InvalidOption>::value,
                  "expect option class instance"
                  ", e.g., RelTol, AbsTol, InitialStep, MaxStep, MaxNumSteps");
  }

  template <typename klass, typename... Args>
  constexpr void _setlist(klass, real_type value, Args&&... args) {
    return _setlist(opt_t<klass> { value }, std::forward<Args>(args)...);
  }

  template <typename klass, typename R, typename... Args>
  constexpr void _setlist(option_t<klass, R>&& opt, Args&&... args) {
    _set(std::forward<option_t<klass, R>>(opt));
    _setlist(std::forward<Args>(args)...);
  }
  template <typename... Args>
  constexpr void _setlist(const odeopts_t<real_type>& _other, Args&&... args) {
    *this = _other;
    _setlist(std::forward<Args>(args)...);
  }
  constexpr void _setlist() {}

  // handle any non-option type: _set will fail
  template <typename ExpectOption>
  constexpr void _setlist(ExpectOption) { _set(opt_t<ExpectOption>{0.0}); }

};

//-----------------------------------------------------------------------------

template <typename T>
std::ostream& operator<<(std::ostream& out, const options::odeopts_t<T>& opts) {
  return out << "{rtol=" << opts.rtol << ", atol=" << opts.atol
             << ", h0=" << opts.h0 << ", hmax=" << opts.hmax
             << ", maxsteps=" << opts.maxsteps << '}';
}

//-----------------------------------------------------------------------------
} // namespace options
//-----------------------------------------------------------------------------

/** \defgroup odeint_opts Options to ODE solvers

    If `ode` defines an ODE solver type, use as

    ```
    auto opts = ode::make_options(AbsTol = 1e-6, RelTol = 1e-3);
    opts.set(MaxStep = 1, InitialStep = 0);
    ```

    \sa VC::odeint::options::odeopts_t
    @{
 */

inline const options::rtol_class RelTol;    ///< set relative tolerance
inline const options::atol_class AbsTol;    ///< set absolute tolerance
inline const options::h0_class InitialStep; ///< set initial step size (if >0)
inline const options::hmax_class MaxStep;   ///< set maximum step size
inline const options::maxnsteps_class MaxNumSteps; ///< set maximum # of steps

/// @}

//=============================================================================
} // namespace odeint
} // namespace VC
//=============================================================================
#endif // VC_ODEINT_OPTIONS_HH
