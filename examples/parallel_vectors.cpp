#include <tatooine/parallel_vectors.h>
#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/differentiated_field.h>
#include <tatooine/field_operations.h>

using namespace tatooine;
auto main() -> int {
  auto v = analytical::numerical::abcflow{};
  write(
      parallel_vectors(v, diff(v, 1e-10) * v,
                       linspace{-3.0, 3.0 + 1e-6, 50},
                       linspace{-3.0, 3.0 + 1e-5, 50},
                       linspace{-3.0, 3.0 + 1e-4, 50}),
      "pv_abc.vtk");
}
