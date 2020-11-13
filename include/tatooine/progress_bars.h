#ifndef TATOOINE_PROGRESS_BARS_H
#define TATOOINE_PROGRESS_BARS_H
//==============================================================================
#include <indicators/indeterminate_progress_bar.hpp>
#include <indicators/block_progress_bar.hpp>
#include <concepts>
//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Indicator>
concept indicator_with_progress = requires(Indicator indicator, double p) {
 indicator.set_progress(p);
};

template <typename Indicator>
struct indicator_msg {
  Indicator& indicator;
  auto operator=(char const* msg) -> indicator_msg& {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  auto operator=(std::string const& msg) -> indicator_msg& {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  auto set_text(std::string const& text) -> void {
    indicator.set_option(indicators::option::PostfixText{text});
  }
  auto mark_as_completed() -> void { indicator.mark_as_completed(); }
};
template <typename Indicator>
struct progress_indicator_wrapper {
  Indicator& indicator;
  double& progress;
  auto operator=(char const* msg) -> progress_indicator_wrapper& {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  auto operator=(std::string const& msg) -> progress_indicator_wrapper& {
    indicator.set_option(indicators::option::PostfixText{msg});
    return *this;
  }
  auto set_text(std::string const& text) -> void {
    indicator.set_option(indicators::option::PostfixText{text});
  }
};
//==============================================================================
template <typename F, typename... Args>
  requires
    std::is_invocable_v<F, Args...> ||
    std::is_invocable_v<F, indicator_msg<indicators::IndeterminateProgressBar>,
                   Args...>
auto indeterminate_progress_bar(F&& f, Args&&... args)
  -> decltype(auto) {
  indicators::IndeterminateProgressBar indicator{
      indicators::option::BarWidth{20},
      indicators::option::Start{"▶"},
      indicators::option::Fill{" "},
      indicators::option::Lead{"░▒▓▒░"},
      indicators::option::End{"◀"},
      indicators::option::ForegroundColor{indicators::Color::green},
      indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};

  std::thread job_completion_thread([&indicator]() {
    while (!indicator.is_completed()) {
      indicator.tick();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
      indicator.tick();
  });

  if constexpr (std::is_invocable_v<F, Args...>) {
    using ret_t = std::invoke_result_t<
        F,  Args...>;
    if constexpr (std::is_same_v<void, ret_t>) {
      f(std::forward<Args>(args)...);
      indicator.mark_as_completed();
      job_completion_thread.join();
    } else {
      decltype(auto) ret = f(std::forward<Args>(args)...);
      indicator.mark_as_completed();
      job_completion_thread.join();
      return ret;
    }
  } else {
    using ret_t = std::invoke_result_t<
        F, indicator_msg<indicators::IndeterminateProgressBar>, Args...>;
    if constexpr (std::is_same_v<void, ret_t>) {
      f(indicator_msg<indicators::IndeterminateProgressBar>{indicator},
        std::forward<Args>(args)...);
      indicator.mark_as_completed();
      job_completion_thread.join();
    } else {
      decltype(auto) ret =
          f(indicator_msg<indicators::IndeterminateProgressBar>{indicator},
            std::forward<Args>(args)...);
      indicator.mark_as_completed();
      job_completion_thread.join();
      return ret;
    }
  }
}
//==============================================================================
template <typename F, typename... Args>
  requires
  std::invocable<F, progress_indicator_wrapper<indicators::BlockProgressBar>,
                   Args...>
auto progress_bar(F&& f, Args&&... args)
  -> decltype(auto) {
  indicators::BlockProgressBar progress_indicator{
      indicators::option::BarWidth{20},
      indicators::option::Start{"▶"},
      indicators::option::End{"◀"},
      indicators::option::ShowElapsedTime{true},
      indicators::option::ShowRemainingTime{true},
      indicators::option::ForegroundColor{indicators::Color::green},
      indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}};
  double progress;
  progress_indicator_wrapper<indicators::BlockProgressBar> wrapper{
      progress_indicator, progress};

  std::thread indicator([&progress_indicator, &progress] {
    while (progress < 1) {
      progress_indicator.set_progress(100 * progress);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    progress_indicator.set_progress(100);
  });

  f(wrapper, std::forward<Args>(args)...);

  indicator.join();
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
