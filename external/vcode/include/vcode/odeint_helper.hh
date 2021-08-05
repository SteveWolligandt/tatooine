#ifndef VC_ODEINT_HELPER_HH
#define VC_ODEINT_HELPER_HH
//=============================================================================
# include <cassert>
# include <utility>      // std::forward
# include <type_traits>  // std::is_*, std::invoke_result
//-----------------------------------------------------------------------------
# include "odeint_evaldy.hh"
//=============================================================================
namespace VC {
namespace odeint {
//=============================================================================

/** Define vector operations such that they can be adapted to own types.

    Specialize this class in case you use your own vector class.
 */
template <typename V>
struct vector_operations_t {
  static_assert(!std::is_same<V, V>::value,
                "missing specialization for vector class V");
};

//-----------------------------------------------------------------------------
namespace detail {
//-----------------------------------------------------------------------------

template <typename T, typename R>
struct helper_t {
  using vec_t = T;
  using real_t = R;

  using maybe_vec = maybe<vec_t>;

  static bool isfinitenorm(const vec_t& _v) {
    if constexpr (std::is_same<vec_t, real_t>::value)
      return std::isfinite(_v);
    else
      return vector_operations_t<vec_t>::isfinitenorm(_v);
  }

  /** Helper: Check norm `_dy(t,y)` unless result is a `maybe_vec`.

      If `_dy` returns a vector (not a `maybe_vec`), generate a new
      function that retuns a `maybe_vec`, which is undefined if the
      result is not finite.

      Note that any `_dy` that returns a `maybe_vec` will be returned
      *unmodified* (w/o checks), because such `_dy` is expected to
      **never** return an infinite result, which is not undefined!
   */
  template <typename DY>
  static auto require_finite(DY&& _dy) {
    using result_t= typename std::invoke_result<DY, real_t, vec_t>::type;

    if constexpr (std::is_same<result_t, maybe_vec>::value) {
#ifndef NDEBUG
      return std::forward<DY>(_dy);
#else
      return [_dy](real_t t, const vec_t& y) {
        auto dy= _dy(t, y);
        assert((dy.undefined() || std::isfinitenorm(dy)) &&
               "expect finite norm for successful evaluation");
        return dy;
      };
#endif
    } else {
      return [_dy](real_t t, const vec_t& y) -> maybe_vec {
        auto dy= _dy(t, y);
        if (!isfinitenorm(dy)) return Failed;
        return dy;
      };
    }
  }
  //-----------------------------------------------------------------------------

  /// Helper: Determine is `Sink(...)` takes `t,y` (value only") or `t,y,dy`.
  template <typename Sink>
  struct pass_value_only_t
      : std::integral_constant<bool,
                               std::is_invocable<Sink, real_t, vec_t>::value> {
    static_assert(std::is_invocable<Sink, real_t, vec_t>::value ||
                      std::is_invocable<Sink, real_t, vec_t, vec_t>::value,
                  "Require sink callable as sink(t,y) or sink(t,y,dy).");
  };

  //-----------------------------------------------------------------------------

  /// Helper: Pass evaluation result to sink.
  template <typename Evaluator, typename Sink>
  struct evaluate_function_into_t {
    Evaluator evaluator;
    Sink sink;

    evaluate_function_into_t(Evaluator&& _evaluator, Sink&& _sink)
        : evaluator(std::forward<Evaluator>(_evaluator)),
          sink(std::forward<Sink>(_sink)) {}

    // TODO: alternative: evaluator always returns tuple (t,...)
    auto operator()(real_t _t) const {
      if constexpr (pass_value_only_t<Sink>::value) {
        return sink(_t, evaluator(_t));
      } else {
        auto [y, dy]= evaluator(_t);
        return sink(_t, y, dy);
      }
    };
  };
  /// Helper: Pass evaluation result to sink.
  template <typename Evaluator, typename Sink>
  static auto evaluate_function_into(Evaluator&& _evaluator, Sink&& _sink) {
    return evaluate_function_into_t<Evaluator, Sink>(
        std::forward<Evaluator>(_evaluator), std::forward<Sink>(_sink));
  }

};  // struct helper_t

//-----------------------------------------------------------------------------
}  // namespace detail
//=============================================================================
}  // namespace odeint
}  // namespace VC
//=============================================================================
#endif  // VC_ODEINT_HELPER_HH
