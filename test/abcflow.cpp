#include "../abcflow.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("abcflow", "[abcflow][abc][symbolic][numerical]") {
  symbolic::abcflow sabc;
  numerical::abcflow nabc;
  for (size_t x = 0; x < 10; ++x) {
    for (size_t y = 0; y < 10; ++y) {
      for (size_t z = 0; z < 10; ++z) {
        vec3 p{x, y, z};
        auto vs = sabc(p, 0);
        auto vn = nabc(p, 0);
        for (size_t i = 0; i < 3; ++i) { CHECK(vs(i) == Approx(vn(i))); }
      }
    }
  }
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
