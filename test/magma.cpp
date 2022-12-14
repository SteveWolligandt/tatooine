#include <tatooine/color_scales/magma.h>
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::color_scales::test {
//==============================================================================
TEST_CASE("magma"){
  magma scale;
  REQUIRE(approx_equal(scale(0), vec{0.001462, 0.000466, 0.013866}));
  REQUIRE(approx_equal(scale(1), vec{0.98705299999999996, 0.99143800000000004,
                                     0.74950399999999995}));
}
//==============================================================================
}  // namespace tatooine::color_scales::test
//==============================================================================
