#include "benchmark.h"
#include <tatooine/gpu/reduce.h>
//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
static void texture_reduction(::benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();  // Stop timers. They will not count until they are
                          // resumed.
    const size_t             width  = state.range(0);
    const size_t             height = state.range(1);
    const std::vector<float> rand_data(width*height, 0.5f);
    const yavin::tex2r32f data_tex{rand_data, width, height};
    state.ResumeTiming();  // And resume timers. They are now counting again.
    gpu::reduce(data_tex, state.range(2), state.range(3));
  }
}
static void texture_reduction_args(::benchmark::internal::Benchmark* b) {
  for (int res = 32; res <= 1024; res *= 2) {
    for (int w = 8; w <= 32; w *= 2) { b->Args({res, res, w, w}); }
  }
}
BENCHMARK(texture_reduction)->Apply(texture_reduction_args);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
