#include <tatooine/tuple.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("tuple", "[tuple]") {
  {
    auto const t = tuple{1, 2.0f};
    t.iterate([i = std::size_t{}](auto const& x) mutable {
      switch (i) {
        case 0:
          REQUIRE(std::same_as<std::decay_t<decltype(x)>, int>);
          REQUIRE(x == 1);
          break;
        case 1:
          REQUIRE(std::same_as<std::decay_t<decltype(x)>, float>);
          REQUIRE(x == 2.0f);
          break;
        default:
          break;
      }
      ++i;
    });
    REQUIRE(t.at<0>() == 1);
    REQUIRE(t.at<1>() == 2.0f);
  }
  {
    auto t = tuple<float, float>{1, 2.0f};
    t.iterate([i = std::size_t{}](auto const& x) mutable {
      switch (i) {
        case 0:
          REQUIRE(std::same_as<std::decay_t<decltype(x)>, float>);
          REQUIRE(x == 1.0f);
          break;
        case 1:
          REQUIRE(std::same_as<std::decay_t<decltype(x)>, float>);
          REQUIRE(x == 2.0f);
          break;
        default:
          break;
      }
      ++i;
    });
    auto const float_arr = t.as_pointer();
    REQUIRE(float_arr[0] == 1.0f);
    REQUIRE(float_arr[1] == 2.0f);
    REQUIRE(t.at<0>() == 1.0f);
    REQUIRE(t.at<1>() == 2.0f);
  }
  {
    auto t   = tuple{1};
    auto ptr = t.as_pointer();
    REQUIRE(ptr[0] == 1);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
