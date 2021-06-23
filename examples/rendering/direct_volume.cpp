#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/field_operations.h>
#include <tatooine/rendering/volume.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  //auto const v   = analytical::fields::numerical::doublegyre{};
  //auto const stv = spacetime(v);
  //auto       vl  = length(stv);
  //auto       discretized_domain =
  //    grid{linspace{0.0,  2.0, 201},
  //         linspace{0.0,  1.0, 101},
  //         linspace{0.0, 10.0, 101}};
  //rendering::interactive(
  //    discretize(vl, discretized_domain, "discretized_length", 0));
  //rendering::interactive(
  //    discretize(Q(stv), discretized_domain, "Q", 0));

  auto const w   = analytical::fields::numerical::abcflow{};
  auto       discretized_domain_2 =
      grid{linspace{-10.0, 10.0, 201},
           linspace{-10.0, 10.0, 201},
           linspace{-10.0, 10.0, 201}};
  rendering::interactive(
      discretize(Q(w), discretized_domain_2, "length", 0));
}
