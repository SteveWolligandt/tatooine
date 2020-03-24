#include "benchmark.h"

#include <tatooine/gpu/reduce.h>
#include <tatooine/random.h>
//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
static void texture_reduction(::benchmark::State& state) {
  auto         seed   = std::random_device{}();
  std::mt19937 eng{seed};


  for (auto _ : state) {
    state.PauseTiming();  // Stop timers. They will not count until they are
                          // resumed.
    const size_t             width  = state.range(2);
    const size_t             height = state.range(2);
    const std::vector<float> rand_data =
        random_uniform_vector<float>(width * height, 0.0f, 1.0f, eng);
    const yavin::tex2r32f data_tex{rand_data, width, height};
    state.ResumeTiming();  // And resume timers. They are now counting again.
    gpu::reduce(data_tex, state.range(0), state.range(1));
  }
}
static void texture_reduction_args(::benchmark::internal::Benchmark* b) {
  for (int k = 32; k <= 1024; k *= 2)
    for (int i = 8; i <= 32; i *= 2)
      for (int j = 8; j <= 32; j *= 2) b->Args({i, j, k});
}
BENCHMARK(texture_reduction)->Apply(texture_reduction_args);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
