#include <tatooine/random.h>
#include <yavin/texture.h>
#include <execution>
#include <numeric>

#include "benchmark.h"
//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
static void cpu_reduction(::benchmark::State& state) {
  auto         seed   = std::random_device{}();
  std::mt19937 eng{seed};


  for (auto _ : state) {
    state.PauseTiming();  // Stop timers. They will not count until they are
                          // resumed.
    const size_t             width = state.range(0);
    const size_t             height = state.range(1);
    const std::vector<float> rand_data =
        random_uniform_vector<float>(width * height, 0.0f, 1.0f, eng);
    const yavin::tex2r32f data_tex{rand_data, width, height};
    state.ResumeTiming();  // And resume timers. They are now counting again.
    auto downloaded_data = data_tex.download_data();
    std::reduce(std::execution::par, begin(downloaded_data),
                end(downloaded_data), 0.0f);
  }
}
static void cpu_reduction_args(::benchmark::internal::Benchmark* b) {
  for (int i = 32; i <= 1024; i *= 2) b->Args({i,i});
}
BENCHMARK(cpu_reduction)->Apply(cpu_reduction_args);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
