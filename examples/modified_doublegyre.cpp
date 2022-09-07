#include <tatooine/analytical/modified_doublegyre.h>
#include <tatooine/linspace.h>
#include <tatooine/rendering/interactive.h>
//==============================================================================
using namespace tatooine;
using analytical::numerical::modified_doublegyre;
//==============================================================================
auto main() -> int {
  auto v = modified_doublegyre{};
  auto lcs = v.lagrangian_coherent_structure(0);
  auto discretized_lcs = line2{};

  for (auto const t : linspace{0.0, 30.0, 1000}) {
    discretized_lcs.push_back(lcs(t));
  }
  discretized_lcs.write("modified_doublegyre.lcs.vtp");
}
