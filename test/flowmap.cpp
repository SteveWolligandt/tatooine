#include <tatooine/flowmap.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/doublegyre.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("flowmap_doublegyre", "[flowmap][doublegyre][dg]") {
  numerical::doublegyre dg;
  flowmap flowmap_dg{dg, integration::vclibs::rungekutta43<double, 2>{}, 10.0};
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
