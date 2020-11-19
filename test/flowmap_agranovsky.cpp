#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/flowmap_agranovsky.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("flowmap_agranovsky", "[flowmap][agranovsky]") {
  analytical::fields::numerical::doublegyre v;
  auto                                      fm  = flowmap(v);
  double                                    tau = 10;
  double                                    delta_t = 0.5;
  size_t                                    res_x = 20, res_y = 10;
  flowmap_agranovsky fm_agr{v,          0,          tau,   delta_t,
                            vec2{0, 0}, vec2{2, 1}, res_x, res_y};
  fm_agr.write();
  REQUIRE(distance(fm_agr.evaluate_full_forward({1, 0.5}),
                   fm({1, 0.5}, 0, tau)) < 1e-4);
  REQUIRE(distance(fm_agr.evaluate_full_backward({1, 0.5}),
                   fm({1, 0.5}, tau, -tau)) < 1e-4);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
