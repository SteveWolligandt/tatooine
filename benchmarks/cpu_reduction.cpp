//#include <tatooine/random.h>
#include <tatooine/gl/texture.h>
#include <execution>
#include <numeric>

#include "benchmark.h"
//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
static void cpu_reduction(::benchmark::State& state) {
  for (auto _ : state) {
    state.PauseTiming();  // Stop timers. They will not count until they are
                          // resumed.
    const size_t             width  = state.range(0);
    const size_t             height = state.range(1);
    const std::vector<float> rand_data(width * height, 0.5f);
    //= random::uniform_vector<float>(width * height, 0.0f, 1.0f, eng);
    const gl::tex2r32f data_tex{rand_data, width, height};
    state.ResumeTiming();  // And resume timers. They are now counting again.

    auto downloaded_data = data_tex.download_data();
    std::reduce(std::execution::par, begin(downloaded_data),
                end(downloaded_data), 0.0f);
  }
}
static void cpu_reduction_args(::benchmark::internal::Benchmark* b) {
  for (int res = 32; res <= 1024; res *= 2) { b->Args({res, res}); }
}
BENCHMARK(cpu_reduction)->Apply(cpu_reduction_args);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
