#include <benchmark/benchmark.h>

#define TATBENCH(NAME)                  \
  void NAME(::benchmark::State& state); \
  BENCHMARK(NAME);                      \
  void NAME(::benchmark::State& state)

#define TATBENCH_MEASURE for (auto _ : state)
