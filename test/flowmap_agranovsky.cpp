#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/flowmap_agranovsky.h>
#include <tatooine/numerical_flowmap.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("flowmap_agranovsky", "[flowmap][agranovsky]") {
  analytical::numerical::doublegyre v;
  auto                                      fm = flowmap(v);
  fm.use_caching(false);
  double const       t0      = 00;
  double const       tau     = 10;
  double const       delta_t = 1;
  size_t const       res_x = 400, res_y = 200;
  flowmap_agranovsky fm_agr{v,          0,          tau,   delta_t,
                            vec2{0, 0}, vec2{2, 1}, res_x, res_y};
  fm_agr.write();
  CHECK(distance(fm_agr.evaluate_full_forward({1, 0.5}),
                 fm({1, 0.5}, t0, tau)) < 1e-4);
  CHECK(distance(fm_agr.evaluate_full_backward({1, 0.5}),
                 fm({1, 0.5}, t0 + tau, -tau)) < 1e-4);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
