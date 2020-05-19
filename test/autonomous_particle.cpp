#include <tatooine/autonomous_particle.h>

#include <catch2/catch.hpp>
#include <tatooine/doublegyre.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("autonomnous_particle0", "[autonomous_particle]") {
  numerical::doublegyre v;
  autonomous_particle   p0{vec{0.1, 0.1}, 0, 0.1};
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
