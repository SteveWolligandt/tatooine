#include "benchmark.h"
#include <tatooine/line.h>

//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
TATBENCH(line_sampling) {
  vec v{1.0, 2.0};
  parameterized_line<double, 2> l{
      {vec{0.0, 0.0}, 0.0}, {vec{1.0, 1.0}, 1.0}, {vec{2.0, 0.0}, 2.0}};
  linspace ts(0.0, 2.0, 1000);
  for (auto _ : state) { l.resample(ts); }
}
// Register the function as a benchmark
BENCHMARK(line_sampling);
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
