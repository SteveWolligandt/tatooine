#include <tatooine/abcflow.h>
#include <tatooine/doublegyre.h>
#include <tatooine/duffing_oscillator.h>
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
    numerical::doublegyre v2;
    spacetime_field       v{v2};
    flowexplorer::work(argc, argv, v);
  } else if (fieldname == "duffing") {
    numerical::duffing_oscillator v2{0.5, 0.5, 0.5};
    spacetime_field               v{v2};
    flowexplorer::work(argc, argv, v);
  } else if (fieldname == "abc") {
    numerical::abcflow v;
    flowexplorer::work(argc, argv, v);
  }
}
