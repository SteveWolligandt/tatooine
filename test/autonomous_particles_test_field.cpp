#include <tatooine/analytical/fields/numerical/autonomous_particles_test.h>
#include <tatooine/flowmap_gradient_central_differences.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::analytical::fields::numerical::test {
//==============================================================================
TEST_CASE("autonomous_particles_test_field_flowmap",
          "[autonomous_particles_test][flowmap]") {
  autonomous_particles_test v;
  [[maybe_unused]] auto fma   = flowmap(v, tag::analytical);
  [[maybe_unused]] auto fmaga = diff(fma, tag::analytical);
  [[maybe_unused]] auto fmagc = diff(fma, tag::central);
  [[maybe_unused]] auto fmn   = flowmap(v, tag::numerical);
  [[maybe_unused]] auto fmngc = diff(fmn, tag::central);
}
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical::test
//==============================================================================
