#include "../doublegyre.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("doublegyre", "[doublegyre]") {
  numerical::doublegyre ndg;
  symbolic::doublegyre  adg;

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
  test(adg, vec{0.0, 0.0});
  test(adg, vec{0.5, 0.5});
  test(adg, vec{1.5, 0.5});
  test(adg, vec{2.0, 1.0});
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
