#include <tatooine/flowmap.h>
#include <tatooine/doublegyre.h>
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("flowmap_doublegyre", "[flowmap][doublegyre][dg]") {
  numerical::doublegyre dg;
  auto                  flowmap_dg =
      flowmap{dg, ode::boost::rungekuttafehlberg78<double, 2>{}, 10.0};
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
