#ifndef VC_ODEINT_GENERATOR_HH
#define VC_ODEINT_GENERATOR_HH
//=============================================================================
# include <cassert>
# include <type_traits> // std::is_*
//=============================================================================
namespace VC {
namespace odeint {
namespace generator {
//=============================================================================

enum class tag_t {
  n_uniform_steps,
  uniform_delta_steps,
  time_range,
  indexed_time_range
};

template <tag_t Tag, typename R, typename Arg = void>
struct generator_t {
  static_assert(Tag != Tag, "missing specialization");
};

//-----------------------------------------------------------------------------

/// Generate `n` steps with uniform step size.
template <typename R>
struct generator_t<tag_t::n_uniform_steps, R> {
  using real_type = R;
  static_assert(std::is_floating_point<real_type>::value, "expect floating point");

  real_type t0, t1;
  int ns, i;

  generator_t(int _n) : ns(_n - 1), i(0) { assert(_n >= 1); }

  void initialize(real_type _t0, real_type _t1) {
    assert(ns > 0);
    assert(std::isfinite(_t0) && std::isfinite(_t1));
    t0= _t0;
    t1= _t1;
    i= 0;
  }
  bool has_next() const { return i <= ns; }
  real_type next() {
    real_type alpha= real_type(i++) / real_type(ns);
    return (1 - alpha) * t0 + alpha * t1;
  }
};

//-----------------------------------------------------------------------------

/// Generate steps with uniform step size `delta`.
template <typename R>
struct generator_t<tag_t::uniform_delta_steps, R> {
  using real_type = R;
  static_assert(std::is_floating_point<real_type>::value, "expect floating point");

  real_type t1, t, delta;
  generator_t(real_type _delta) : delta(_delta) { assert(delta > 0); }

  void initialize(real_type _t0, real_type _t1) {
    assert(std::isfinite(_t0) || std::isfinite(_t1));
    t= _t0;
    t1= _t1;
    if (_t1 - _t0 < 0) delta= -delta;
  }
  bool has_next() const { return delta > 0 ? t <= t1 : t >= t1; }
  real_type next() {
    real_type tp= t;
    t+= delta;
    return tp;
  }
};

//-----------------------------------------------------------------------------

/// Generate steps at times given by a range `begin, end`.
template <typename R, typename Iterator>
struct generator_t<tag_t::time_range, R, Iterator> {
  using real_type = R;
  using iterator_t = Iterator;
  static_assert(std::is_floating_point<real_type>::value, "expect floating point");

  Iterator ii;
  const Iterator end;

  generator_t(Iterator _begin, const Iterator _end) : ii(_begin), end(_end) {}

  void initialize(real_type, real_type) {}  // no checks
  bool has_next() const { return ii != end; }
  real_type next() {
    static_assert(std::is_convertible<decltype(*ii), real_type>::value,
                  "*iterator must be convertible to real_type");
    return static_cast<R>(*ii++);
  }
};

//-----------------------------------------------------------------------------

/// Generate steps at times `f(i)` for `i=0,1,...,n-1`.
template <typename R, typename F>
struct generator_t<tag_t::indexed_time_range, R, F> {
  using real_type = R;

  static_assert(std::is_floating_point<real_type>::value, "expect floating point");
  static_assert(std::is_invocable<F, int>::value,
                "must be able to evaluate real_type F(int)");
  static_assert(std::is_convertible<
                  typename std::invoke_result<F, int>::type, real_type>::value,
                "value F(int) must be convertible to real_type");

  F f;
  int n, i= 0;

  generator_t(F&& _f, int _n) : f(_f), n(_n) {}

  void initialize(real_type, real_type) {}  // no checks
  bool has_next() const { return i < n; }
  real_type next() { return static_cast<R>(f(i++)); }
};

//-----------------------------------------------------------------------------
} // namespace generator
} // namespace odeint
} // namespace VC
//=============================================================================
#endif // VC_ODEINT_GENERATOR_HH
