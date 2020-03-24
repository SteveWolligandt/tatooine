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
    const size_t             size  = state.range(1);
    const std::vector<float> rand_data =
        random_uniform_vector<float>(size, 0.0f, 1.0f, eng);
    const yavin::shaderstoragebuffer<float> data_tex{rand_data};
    state.ResumeTiming();  // And resume timers. They are now counting again.
    gpu::reduce(data_tex, state.range(0));
  }
}
static void buffer_reduction_args(::benchmark::internal::Benchmark* b) {
  for (int i = 8*8; i <= 32*32; i *= 4)
      for (int j = 512; j <= 4096; j *= 2) b->Args({i, j});
}
BENCHMARK(buffer_reduction)->Apply(buffer_reduction_args);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
