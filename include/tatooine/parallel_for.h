#ifndef TATOOINE_PARALLEL_FOR_H
#define TATOOINE_PARALLEL_FOR_H

#include <thread>
#include <vector>

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename F, typename ForwardIt>
inline void parallel_for(
    ForwardIt begin, ForwardIt end, F&& f,
    [[maybe_unused]]size_t num_threads = std::thread::hardware_concurrency()) {
#ifdef NDEBUG
  size_t grainsize = distance(begin, end) / num_threads;
  while (grainsize == 0 && num_threads > 1) {
    --num_threads;
    grainsize = distance(begin, end) / num_threads;
  }
  std::vector<std::thread> threads;
  auto                     start_it = begin;

  for (size_t i = 0; i < num_threads - 1; ++i, advance(start_it, grainsize)) {
    threads.emplace_back(
        [&f, begin = start_it, end = next(start_it, grainsize)]() {
          for (auto it = begin; it != end; ++it) { f(*it); }
        });
  }
  threads.emplace_back([&f, begin = start_it, end]() {
    for (auto it = begin; it != end; ++it) { f(*it); }
  });

  for (auto& t : threads) { t.join(); }
#else
  for (auto it = begin; it != end; ++it) { f(*it); }
#endif
}

//------------------------------------------------------------------------------
template <typename F, typename Range>
inline void parallel_for(
    Range&& range, F&& f,
    size_t num_threads = std::thread::hardware_concurrency()) {
  parallel_for(begin(range), end(range), std::forward<F>(f), num_threads);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
