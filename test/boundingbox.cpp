#include <tatooine/boundingbox.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("boundingbox_ray_intersection", "[boundingbox][ray][intersection]") {
  boundingbox bb{vec{-1.0, -1.0}, vec{1.0, 1.0}};
  {
    ray  r{vec{0.0, 2.0}, vec{0.0, -1.0}};
    auto i = bb.check_intersection(r);
    REQUIRE(i);
    REQUIRE(i->position == vec{0.0, 1.0});
  }
  {
    ray  r{vec{0.0, -2.0}, vec{0.0, 1.0}};
    auto i = bb.check_intersection(r);
    REQUIRE(i);
    REQUIRE(i->position == vec{0.0, -1.0});
  }
  {
    ray  r{vec{-2.0, -1.0}, vec{1.0, 1.0}};
    auto i = bb.check_intersection(r);
    REQUIRE(i);
    REQUIRE(i->position == vec{-1.0, 0.0});
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
