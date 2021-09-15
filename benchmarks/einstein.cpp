#include <tatooine/einstein_notation.h>
#include <tatooine/tensor.h>

#include "benchmark.h"
//==============================================================================
namespace tatooine::benchmark {
//==============================================================================
TATBENCH(einstein_mulitplication) {
  using namespace einstein_notation;
  auto const A = mat<double, 300, 100>{random::uniform<double>{}};
  auto const B = mat<double, 100, 200>{random::uniform<double>{}};
  auto const C = mat<double, 200, 100>{random::uniform<double>{}};
  auto       D = mat<double, 300, 100>{random::uniform<double>{}};
  TATBENCH_MEASURE { D(i, l) = A(i, j) * B(j, k) * C(k, l); }
}
//==============================================================================
TATBENCH(matrix_multiplication1) {
  auto const A = mat<double, 300, 100>{random::uniform<double>{}};
  auto const B = mat<double, 100, 200>{random::uniform<double>{}};
  auto const C = mat<double, 200, 100>{random::uniform<double>{}};
  auto       D = mat<double, 300, 100>{random::uniform<double>{}};
  TATBENCH_MEASURE { D = (A * B) * C; }
}
//==============================================================================
TATBENCH(matrix_multiplication2) {
  auto const A = mat<double, 300, 100>{random::uniform<double>{}};
  auto const B = mat<double, 100, 200>{random::uniform<double>{}};
  auto const C = mat<double, 200, 100>{random::uniform<double>{}};
  auto       D = mat<double, 300, 100>{random::uniform<double>{}};
  TATBENCH_MEASURE { D = A * (B * C); }
}
//==============================================================================
}  // namespace tatooine::benchmark
//==============================================================================
