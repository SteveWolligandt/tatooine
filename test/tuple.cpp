#include <tatooine/tuple.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("tuple", "[tuple]") {
  auto const t = tuple{1, 1.0f};
  t.iterate([i = std::size_t{}](auto const& x) mutable {
    switch (i) {
      case 0:
        REQUIRE(std::same_as<std::decay_t<decltype(x)>, int>);
        break;
      case 1:
        REQUIRE(std::same_as<std::decay_t<decltype(x)>, float>);
        break;
      default:
        break;
    }
    ++i;
  });
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
