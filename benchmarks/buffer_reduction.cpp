#include "benchmark.h"

#include <tatooine/gpu/reduce.h>
#include <tatooine/random.h>
//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
static void buffer_reduction(::benchmark::State& state) {
  auto         seed   = std::random_device{}();
  std::mt19937 eng{seed};

  for (auto _ : state) {
    state.PauseTiming();  // Stop timers. They will not count until they are
                          // resumed.
    const size_t             width  = state.range(0);
    const size_t             height  = state.range(1);
    const std::vector<float> rand_data =
        random::uniform_vector<float>(width * height, 0.0f, 1.0f, eng);
    const gl::shaderstoragebuffer<float> data_tex{rand_data};
    state.ResumeTiming();  // And resume timers. They are now counting again.

    gpu::reduce(data_tex, state.range(2) * state.range(3));
  }
}
static void buffer_reduction_args(::benchmark::internal::Benchmark* b) {
  for (int res = 32; res <= 1024; res *= 2) {
    for (int w = 8; w <= 32; w *= 2) { b->Args({res, res, w, w}); }
  }
}
BENCHMARK(buffer_reduction)->Apply(buffer_reduction_args);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
