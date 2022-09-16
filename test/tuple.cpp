#include <tatooine/tuple.h>

#include <catch2/catch_test_macros.hpp>
#include <tatooine/type_traits.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("tuple", "[tuple]") {
  SECTION("int float") {
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
  SECTION("float float") {
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
  SECTION("pointer"){
    auto t   = tuple{1};
    auto ptr = t.as_pointer();
    REQUIRE(ptr[0] == 1);
  }
  SECTION("indexing"){
    auto t = tuple{3, 2, 1};
    SECTION("at"){
      REQUIRE(t.at<0>() == 3);
      REQUIRE(t.at<1>() == 2);
      REQUIRE(t.at<2>() == 1);
    }
    SECTION("get") {
      REQUIRE(get<0>(t) == 3);
      REQUIRE(get<1>(t) == 2);
      REQUIRE(get<2>(t) == 1);
    }
  }
  SECTION("concatenation") {
    using tuple_type1 = tuple<int, float>;
    using tuple_type2 = tuple<double, char>;
    using concatenated_tuple = tuple_concat_types<tuple_type1, tuple_type2>;
    REQUIRE(is_same<concatenated_tuple::type_at<0>, int>);
    REQUIRE(is_same<concatenated_tuple::type_at<1>, float>);
    REQUIRE(is_same<concatenated_tuple::type_at<2>, double>);
    REQUIRE(is_same<concatenated_tuple::type_at<3>, char>);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
