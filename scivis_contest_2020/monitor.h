#ifndef TATOOINE_SCIVIS_CONTEST_2020_MONITOR_H
#define TATOOINE_SCIVIS_CONTEST_2020_MONITOR_H
//==============================================================================
#include <thread>
#include <chrono>
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
template <typename F, typename Prog>
void monitor(F&& f, Prog&& prog, std::string const& msg) {
  using timer = std::chrono::high_resolution_clock;
  bool               done = false;
  std::thread        m{[&] {
    while (!done) {
      std::cerr << msg << ": " << prog() * 100 << "%        \r";
      std::this_thread::sleep_for(std::chrono::milliseconds{200});
    }
  }};
  auto const before = timer::now();
  f();
  auto const after = timer::now();
  done = true;
  m.join();
  auto const t =
      std::chrono::duration_cast<std::chrono::seconds>(after - before).count();
  std::cerr << msg << ": done (" << t << " seconds)\n";
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
#endif
