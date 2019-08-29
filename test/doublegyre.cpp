#include "../doublegyre.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("doublegyre", "[doublegyre][dg][symbolic][numerical]") {
  numerical::doublegyre ndg;
  symbolic::doublegyre  sdg;
  for (size_t x = 0; x < 10; ++x) {
    for (size_t y = 0; y < 10; ++y) {
      for (size_t t = 0; t < 10; ++t) {
        vec2 p{x / 9 * 2, y / 9};
        auto vs = sdg(p, t);
        auto vn = ndg(p, t);
        for (size_t i = 0; i < 2; ++i) {
          REQUIRE(std::abs(vs(i) - vn(i)) < 1e-5);
        }
      }
    }
  }

  auto test = [&](const auto& dg, const auto& x) {
    const auto v = dg(x, 0);
    INFO("dg({" << x(0) << ", " << x(1) << "}, " << 0 << ") = {" << v(0) << ", "
                << v(1) << "}");
    CHECK(std::abs(v(0)) < 1e-6);
    CHECK(std::abs(v(1)) < 1e-6);
  };

  test(ndg, vec{0.0, 0.0});
  test(ndg, vec{0.5, 0.5});
  test(ndg, vec{1.5, 0.5});
  test(ndg, vec{2.0, 1.0});
  test(sdg, vec{0.0, 0.0});
  test(sdg, vec{0.5, 0.5});
  test(sdg, vec{1.5, 0.5});
  test(sdg, vec{2.0, 1.0});
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
