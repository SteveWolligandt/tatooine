#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/field_operations.h>
#include <tatooine/rendering/volume.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto const w   = analytical::numerical::abcflow{};
  auto       discretized_domain_2 =
      rectilinear_grid{linspace{-10.0, 10.0, 201},
                       linspace{-10.0, 10.0, 201},
                       linspace{-10.0, 10.0, 201}};
  rendering::interactive(
      discretize(Q(w), discretized_domain_2, "length", 0));
}
