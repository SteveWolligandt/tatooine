#ifndef VC_ODEINT_OUTPUT_HH
#define VC_ODEINT_OUTPUT_HH
//=============================================================================
# include <cassert>
# include <utility>      // std::forward
# include <type_traits>  // std::is_*, std::invoke_result
//=============================================================================
namespace VC {
namespace odeint {
//=============================================================================
namespace detail {
//-----------------------------------------------------------------------------

template <typename Output, typename... Args>
struct output_pipeline_t {
  using head_t = Output;
  using tail_t = output_pipeline_t<Args...>;

  constexpr output_pipeline_t(Output&& _output, Args&&... _args)
      : head(std::forward<Output>(_output)),
        tail(std::forward<Args>(_args)...) {}

  template <typename Stepper>
  evstate_t operator()(const Stepper& _stepper) {
    const auto result = _stepper.evaluate_output(head);

    if (result == evstate_t::OK) {
      return tail(_stepper);
    }
    else {
      return result;
    }
  }

  template <typename Arg>
  constexpr auto operator>>(Arg&& arg) const {
    return output_pipeline_t<head_t>{ head_t { head } } >>
        (tail >> std::forward<Arg>(arg));
  }

  head_t head;
  tail_t tail;
};

//-----------------------------------------------------------------------------

template <typename Output>
struct output_pipeline_t<Output> {
  using head_t = Output;

  constexpr output_pipeline_t(Output&& _output)
      : head(std::forward<Output>(_output)) {}

  template <typename Stepper>
  evstate_t operator()(const Stepper& _stepper) {
    return _stepper.evaluate_output(head);
  }

  template <typename Arg>
  constexpr auto operator>>(Arg&& arg) const {
    return output_pipeline_t<Output, Arg> {
      head_t { head }, std::forward<Arg>(arg)
     };
  }

  head_t head;
};

//-----------------------------------------------------------------------------

// TODO: use proper specialization ! (operator>> outside ?

// Note: Cannot specialize in ode_tr scope!
struct empty_output_pipeline_t {
  template <typename Stepper>
  constexpr evstate_t operator()(const Stepper&) {
    return evstate_t::OK;
  }
  template <typename Arg>
  constexpr auto operator>>(Arg&& arg) const {
      return output_pipeline_t<Arg>(std::forward<Arg>(arg));
  }
};

//-----------------------------------------------------------------------------

template <typename Arg0, typename... Args>
constexpr auto output(Arg0&& arg0, Args&&... args) {
  if constexpr (sizeof...(args) == 0) {
    return output_pipeline_t<Arg0> { std::forward<Arg0>(arg0) };
  }
  else {
    return output_pipeline_t<Arg0, Args...> {
      std::forward<Arg0>(arg0), std::forward<Args>(args)...
    };
  }
}

//-----------------------------------------------------------------------------

template <typename T, typename R>
struct output_t {
  using vec_t= T;
  using real_t= R;

  using helper = detail::helper_t<vec_t, real_t>;
  using spline_t = hermite::spline_t<vec_t, real_t>;

  template <typename Generator, typename Sink>
  struct dense_output_t {
    Generator  generator; // generates next samples t
    real_t     t;         // current t, will be consumed next
    int        n = -1;    // number of consumed samples (<0 if uninitialized)
    const Sink sink;      // consumer

    constexpr dense_output_t(Generator&& _generator, Sink&& _sink)
        : generator(_generator), sink(_sink) {}

    template <typename Stepper>
    void initialize_once(const Stepper& st) {
      if (n < 0) {
        n = 0;
        generator.initialize(st.t, st.tfinal);
        assert(generator.has_next());
        t = generator.next();
      }
    }

    template <typename Stepper>
    auto operator()(const Stepper& st) {
      initialize_once(st);

      if (st.is_inside(t)) {
        const auto push_interp_at = st.hermite_interpolator_into(sink);

        do {
          push_interp_at(t);

          ++n;

          if (!generator.has_next())
            break;

          t = generator.next();

        } while (st.is_inside(t));
      }

      return AcceptStep;
    }
  };

  template <typename Sink>
  struct sink_t {
    const Sink sink;

    constexpr sink_t(Sink&& _sink) : sink(_sink) {}

    template <typename Stepper>
    constexpr auto operator()(const Stepper& st) const {
      if constexpr (helper::template pass_value_only_t<Sink>::value) {
        sink(st.tnew, st.ynew);
      }
      else {
        sink(st.tnew, st.ynew, st.dynew);
      }
      return AcceptStep;
    }
  };
}; // struct output_t

//-----------------------------------------------------------------------------
}  // namespace detail
//-----------------------------------------------------------------------------

/** Placeholder for initiation an output "pipeline".

    Example: The expression

    ```
    Output >> preidcate([](...){...}( >> sink([](...){...})
    ```

    yields the same result (an output pipeline) as

    ```
    output(preidcate([](...){...}, sink([](...){...})
    ```

    \sa output()
 */
static constexpr detail::empty_output_pipeline_t Output {};

/** Define output pipeline.

    Example:

    ```
    output(preidcate([](...){...}, ..., sink([](...){...})
    ```
 */
template <typename... Args>
constexpr auto output(Args&&... args) {
  return detail::output(std::forward<Args>(args)...);
}

//=============================================================================
}  // namespace odeint
}  // namespace VC
//=============================================================================
#endif  // VC_ODEINT_OUTPUT_HH
