#include "benchmark.h"
#include <tatooine/line.h>

//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
static void line_sampling(::benchmark::State& /*state*/) {
  vec v{1.0, 2.0};
  //line<double, 2>{{vec{0.0, 0.0}, 0.0}, {vec{1.0, 1.0}, 1.0}, {vec{2.0, 0.0}, 2.0}};
}
BENCHMARK(line_sampling);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
// Register the function as a benchmark
