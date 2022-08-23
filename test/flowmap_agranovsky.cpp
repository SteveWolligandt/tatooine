#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/agranovsky_flowmap_discretization.h>
#include <tatooine/numerical_flowmap.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("flowmap_agranovsky", "[flowmap][agranovsky]") {
  auto v = analytical::numerical::doublegyre {};
  auto fm = flowmap(v);
  fm.use_caching(false);
  auto const        t0      = real_number{0};
  auto const        tau     = real_number{10};
  auto const        delta_t = 1;
  std::size_t const res_x = 40, res_y = 20;
  auto              fm_agra = agranovsky_flowmap_discretization2{
      fm, t0, tau, delta_t, vec2{0, 0}, vec2{2, 1}, res_x, res_y};
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
