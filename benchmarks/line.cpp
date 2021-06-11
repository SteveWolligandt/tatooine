#include <tatooine/line.h>
#include "benchmark.h"
//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
TATBENCH(line_sampling) {
  parameterized_line<double, 2, interpolation::cubic> l{
      {vec{0.0, 0.0}, 0.0}, {vec{1.0, 1.0}, 1.0}, {vec{2.0, 0.0}, 2.0}};
  linspace ts(0.0, 2.0, 100);
  TATBENCH_MEASURE { l.resample(ts); }
}
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
