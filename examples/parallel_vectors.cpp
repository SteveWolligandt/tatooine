#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/analytical/numerical/tornado.h>
#include <tatooine/differentiated_field.h>
#include <tatooine/field_operations.h>
#include <tatooine/parallel_vectors.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto calc_abc_flow() {
  auto       v        = analytical::numerical::abcflow{};
  auto       J        = diff(v, 1e-10);
  auto       a        = J * v;
  auto const pv_lines = parallel_vectors(
      v, a, linspace{-3.0, 3.0 + 1e-6, 100}, linspace{-3.0, 3.0 + 1e-5, 100},
      linspace{-3.0, 3.0 + 1e-4, 100}, execution_policy::parallel);
  write(pv_lines, "pv_abc.vtp");
}
////==============================================================================
// auto calc_tornado_flow() {
//   auto v = analytical::numerical::tornado{};
//   write(parallel_vectors(
//             v, diff(v, 1e-10) * v,
//             linspace{-1.0, 1.0 + 1e-6, 300},
//             linspace{-1.0, 1.0 + 1e-5, 300},
//             linspace{0.0, 1.0 + 1e-4, 150},
//             execution_policy::parallel),
//         "pv_tornado.vtp");
// }
//==============================================================================
auto main() -> int {
  calc_abc_flow();
  // calc_tornado_flow();
}
