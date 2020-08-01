#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/duffing_oscillator.h>
#include <tatooine/spacetime_field.h>

#include "window.h"
//==============================================================================
namespace tatooine::flowexplorer {
//==============================================================================
template <typename V, typename Real, size_t N>
void work(int argc, char** argv, const field<V, Real, N, N>& v) {
  window       w{v};
}
//==============================================================================
}  // namespace tatooine::flowexplorer
//==============================================================================
auto main(int argc, char** argv) -> int {
  using namespace tatooine;
  const std::string fieldname = argv[1];
  if (fieldname == "dg") {
    analytical::fields::numerical::doublegyre v2;
    spacetime_field       v{v2};
    flowexplorer::work(argc, argv, v);
  } else if (fieldname == "duffing") {
    analytical::fields::numerical::duffing_oscillator v2{0.5, 0.5, 0.5};
    spacetime_field               v{v2};
    flowexplorer::work(argc, argv, v);
  } else if (fieldname == "abc") {
    analytical::fields::numerical::abcflow v;
    flowexplorer::work(argc, argv, v);
  }
}
