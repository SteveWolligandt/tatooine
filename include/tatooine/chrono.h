#ifndef TATOOINE_CHRONO_H
#define TATOOINE_CHRONO_H

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename F, typename... Param>
auto measure(F&& f, Param&&... param) {
  using time  = std::chrono::high_resolution_clock;
  using f_return_type = decltype(f(std::forward<Param>(param)...));

  auto before = time::now();

  if constexpr (std::is_same_v<f_return_type, void>) {
    f(std::forward<Param>(param)...);
    return time::now() - before;

  } else {
    decltype(auto) ret   = f(std::forward<Param>(param)...);
    auto           duration = time::now() - before;
    return std::pair<decltype(duration), decltype(ret)>{duration, ret};
  }
}

//------------------------------------------------------------------------------
template <typename... Durations, typename DurationIn>
auto break_down_durations(DurationIn d) {
  using namespace std::chrono;
  std::tuple<Durations...> retval;
  (((std::get<Durations>(retval) = duration_cast<Durations>(d)),
    (d -= duration_cast<DurationIn>(std::get<Durations>(retval)))),
   ...);
  return retval;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
