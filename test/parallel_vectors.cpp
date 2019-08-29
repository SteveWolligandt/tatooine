#include <catch2/catch.hpp>

#include "../parallel_vectors.h"
#include "../derived_field.h"
#include "../doublegyre.h"
#include "../spacetime.h"

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("parallel_vectors", "[parallel_vectors][pv]") {
  symbolic::doublegyre dg;
  spacetime            stdg{dg};
  auto                 jstdg   = diff(stdg);
  auto                 accstdg = jstdg * stdg;
  parallel_vectors     pv{stdg, accstdg,
                          grid{linspace{0.0, 2.0, 21},
                               linspace{0.0, 1.0, 11},
                               linspace{0.0, 1.0, 11}}};
  write_vtk(pv(), "spacetime_doublegure_pv_lines.vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
