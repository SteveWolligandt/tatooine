#include <tatooine/analytical/numerical/counterexample_sadlo.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/analytical/numerical/modified_doublegyre.h>
#include <tatooine/analytical/numerical/saddle.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/for_loop.h>
#include <tatooine/ftle_field.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/linspace.h>
#include <sstream>

#ifdef Always
#undef Always
#endif
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace tatooine::analytical::numerical;
//==============================================================================
TEST_CASE("ftle_doublegyre", "[ftle][doublegyre][dg]") {
  auto const v   = doublegyre{};
  auto const tau = real_number{10};
  auto       f   = ftle_field{v, tau};
  f.flowmap_gradient().flowmap().use_caching(false);
  f(vec{1.0, 0.5}, 0);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
