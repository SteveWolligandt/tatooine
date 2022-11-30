#ifndef TATOOINE_PROGRESS_BARS_H
#define TATOOINE_PROGRESS_BARS_H
//==============================================================================
#include <concepts>
#include <indicators/block_progress_bar.hpp>
#include <indicators/indeterminate_progress_bar.hpp>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Indicator>
concept indicator_with_progress =
    requires(Indicator indicator, double p) { indicator.set_progress(p); };

template <typename Indicator> struct indicator_msg {
  Indicator &indicator;
  auto operator=(char const *msg) -> indicator_msg & {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  auto operator=(std::string const &msg) -> indicator_msg & {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  auto set_text(std::string const &text) -> void {
    indicator.set_option(indicators::option::PostfixText{text});
  }
  auto mark_as_completed() -> void { indicator.mark_as_completed(); }
};
template <typename Indicator> struct progress_indicator_wrapper {
  Indicator &indicator;
  double &progress;
  //----------------------------------------------------------------------------
  auto operator=(char const *msg) -> progress_indicator_wrapper & {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator=(std::string const &msg) -> progress_indicator_wrapper & {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  //----------------------------------------------------------------------------
  auto set_text(std::string const &text) -> void {
    indicator.set_option(indicators::option::PostfixText{text});
  }
};
//==============================================================================
auto make_default_indeterminate_progress_bar() {
  using namespace indicators;
  return IndeterminateProgressBar{
      option::BarWidth{20},
      option::Start{"▶"},
      option::Fill{" "},
      option::Lead{"░▒▓▒░"},
      option::End{"◀"},
      option::ForegroundColor{Color::white},
      option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
}
//------------------------------------------------------------------------------
auto make_indeterminate_completion_thread(auto &indicator) {
  return std::thread{[&indicator]() {
    while (!indicator.is_completed()) {
      indicator.tick();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    indicator.tick();
  }};
}
//------------------------------------------------------------------------------
template <typename F, typename... Args>
  requires is_invocable<F, Args...>
auto indeterminate_progress_bar(F &&f, Args &&...args) -> decltype(auto) {
  auto indicator = make_default_indeterminate_progress_bar();
  auto job_completion_thread = make_indeterminate_completion_thread(indicator);
  using return_type = invoke_result<F, Args...>;
  if constexpr (is_void<return_type>) {
    f(std::forward<Args>(args)...);
    indicator.mark_as_completed();
    job_completion_thread.join();
  } else {
    decltype(auto) ret = f(std::forward<Args>(args)...);
    indicator.mark_as_completed();
    job_completion_thread.join();
    return ret;
  }
}
//------------------------------------------------------------------------------
template <typename F, typename... Args>
  requires is_invocable<F, indicator_msg<indicators::IndeterminateProgressBar>,
                        Args...> &&
           is_void<invoke_result<
               F, indicator_msg<indicators::IndeterminateProgressBar>, Args...>>
auto indeterminate_progress_bar(F &&f, Args &&...args) {
  auto indicator = make_default_indeterminate_progress_bar();
  auto job_completion_thread = make_indeterminate_completion_thread(indicator);

  f(indicator_msg<indicators::IndeterminateProgressBar>{indicator},
    std::forward<Args>(args)...);
  indicator.mark_as_completed();
  job_completion_thread.join();
}
//------------------------------------------------------------------------------
template <typename F, typename... Args>
  requires is_invocable<F, indicator_msg<indicators::IndeterminateProgressBar>,
                        Args...> &&
           (!is_void<invoke_result<
                F, indicator_msg<indicators::IndeterminateProgressBar>,
                Args...>>)
           auto indeterminate_progress_bar(F &&f, Args &&...args)
               -> decltype(auto) {
  using namespace indicators;
  using return_type =
      invoke_result<F, indicator_msg<indicators::IndeterminateProgressBar>,
                    Args...>;
  auto indicator = make_default_indeterminate_progress_bar();
  auto job_completion_thread = make_indeterminate_completion_thread(indicator);

  decltype(auto) ret =
      f(indicator_msg<indicators::IndeterminateProgressBar>{indicator},
        std::forward<Args>(args)...);
  indicator.mark_as_completed();
  job_completion_thread.join();
  return ret;
}
//==============================================================================
template <typename F, typename... Args>
  requires std::invocable<
               F, progress_indicator_wrapper<indicators::BlockProgressBar>,
               Args...>
auto progress_bar(F &&f, Args &&...args) -> decltype(auto) {
  indicators::BlockProgressBar progress_indicator{
      indicators::option::BarWidth{20},
      // indicators::option::Start{"▶"},
      // indicators::option::End{"◀"},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
      indicators::option::ForegroundColor{indicators::Color::white},
      indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};
  auto progress = double{};
  progress_indicator_wrapper<indicators::BlockProgressBar> wrapper{
      progress_indicator, progress};

  auto job_completion_thread = std::thread{[&progress_indicator, &progress] {
    while (progress < 1) {
      progress_indicator.set_progress(100 * progress);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    progress_indicator.set_progress(100);
  }};

  if constexpr (is_invocable<F, Args...>) {
    using return_type = invoke_result<F, Args...>;
    if constexpr (is_void<return_type>) {
      f(std::forward<Args>(args)...);
      wrapper.progress = 100;
      job_completion_thread.join();
    } else {
      decltype(auto) ret = f(std::forward<Args>(args)...);
      wrapper.progress = 100;
      job_completion_thread.join();
      return ret;
    }
  } else if constexpr (is_invocable<F,
                                    progress_indicator_wrapper<
                                        indicators::BlockProgressBar>,
                                    Args...>) {
    using return_type = invoke_result<
        F, progress_indicator_wrapper<indicators::BlockProgressBar>, Args...>;
    if constexpr (is_void<return_type>) {
      f(wrapper, std::forward<Args>(args)...);
      wrapper.progress = 100;
      job_completion_thread.join();
    } else {
      decltype(auto) ret = f(wrapper, std::forward<Args>(args)...);
      wrapper.progress = 100;
      job_completion_thread.join();
      return ret;
    }
  }
}
//==============================================================================
} // namespace tatooine
//==============================================================================
#endif
