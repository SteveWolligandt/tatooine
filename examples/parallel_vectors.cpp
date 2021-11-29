#include <tatooine/parallel_vectors.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/differentiated_field.h>

using namespace tatooine;
auto main() -> int {
  auto v = analytical::fields::numerical::abcflow{};
  parallel_vectors(v, diff(v, 1e-10) * v, linspace{-1.0, 1.0, 10},
                   linspace{-1.0, 1.0, 10}, linspace{-1.0, 1.0, 10});
}
