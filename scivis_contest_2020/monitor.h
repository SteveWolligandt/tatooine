#ifndef TATOOINE_SCIVIS_CONTEST_2020_MONITOR_H
#define TATOOINE_SCIVIS_CONTEST_2020_MONITOR_H
//==============================================================================
#include <chrono>
#include <thread>

#include "split_durations.h"
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
template <typename F, typename Prog>
void monitor(F&& f, Prog&& prog, std::string const& msg) {
  using timer = std::chrono::high_resolution_clock;
  bool done   = false;
  if constexpr (!std::is_arithmetic_v<std::invoke_result_t<Prog>>) {
    std::cerr << msg << "...\n";
  }
  std::thread m{[&] {
    while (!done) {
      if constexpr (std::is_arithmetic_v<std::invoke_result_t<Prog>>) {
        std::cerr << msg << "... " << prog() * 100 << "%        \r";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{200});
    }
  }};
  auto const  before = timer::now();
  f();
  auto const after = timer::now();
  done             = true;
  m.join();
  auto const duration = after - before;
  std::cerr << msg << "... done (";

  using namespace std::chrono;
  format_duration<days, hours, minutes, seconds, milliseconds>(duration, std::cerr);
  std::cerr << ")\n";
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F>
void monitor(F&& f, std::string const& msg) {
  monitor(
      std::forward<F>(f), [] {}, msg);
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
#endif
