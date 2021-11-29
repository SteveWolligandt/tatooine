#include <tatooine/parallel_vectors.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/differentiated_field.h>

using namespace tatooine;
auto main() -> int {
  auto v = analytical::fields::numerical::abcflow{};
  write_vtk(
      parallel_vectors(v, diff(v, 1e-10) * v, linspace{-3.0, 3.0 + 1e-6, 250},
                       linspace{-3.0, 3.0 + 1e-5, 250},
                       linspace{-3.0, 3.0 + 1e-4, 250}),
      "pv_abc.vtk");
}
