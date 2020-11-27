#include <tatooine/preprocessor.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("preprocessor_num_args", "[preprocessor][num_args]") {
  REQUIRE(TATOOINE_PP_NUM_ARGS() == 0);
  REQUIRE(TATOOINE_PP_NUM_ARGS(3) == 1);
  REQUIRE(TATOOINE_PP_NUM_ARGS(1, 2, 3) == 3);
  REQUIRE(TATOOINE_PP_NUM_ARGS(3, 2, 1) == 3);
  REQUIRE(TATOOINE_PP_NUM_ARGS(3, 1) == 2);
}
//==============================================================================
TEST_CASE("preprocessor_not_equal", "[preprocessor][not_equal]") {
  REQUIRE(TATOOINE_PP_NOT_EQUAL(0, 1));
  REQUIRE_FALSE(TATOOINE_PP_NOT_EQUAL(0, 0));
}
//==============================================================================
TEST_CASE("preprocessor_equal", "[preprocessor][equal]") {
  REQUIRE(TATOOINE_PP_EQUAL(1, 0) == 0);
  REQUIRE(TATOOINE_PP_EQUAL(0, 0) == 1);
  REQUIRE(TATOOINE_PP_EQUAL(1, TATOOINE_PP_NUM_ARGS(1)));
  REQUIRE(TATOOINE_PP_EQUAL(2, TATOOINE_PP_NUM_ARGS(1, 2)));
}
//==============================================================================
TEST_CASE("preprocessor_bool", "[preprocessor][bool]") {
  REQUIRE(TATOOINE_PP_BOOL(0) == 0);
  REQUIRE(TATOOINE_PP_BOOL(1) == 1);
  REQUIRE(TATOOINE_PP_BOOL(2) == 1);
  REQUIRE(TATOOINE_PP_BOOL(3) == 1);
}
//==============================================================================
TEST_CASE("preprocessor_if_else", "[preprocessor][if]") {
  REQUIRE(std::array{0, TATOOINE_PP_IF(0, 1)}.size() == 1);
  REQUIRE(TATOOINE_PP_IF(1, 1) == 1);
  REQUIRE(TATOOINE_PP_IF(1, 2) == 2);
  REQUIRE(TATOOINE_PP_IF_ELSE(1, 2, 3) == 2);
  REQUIRE(TATOOINE_PP_IF_ELSE(0, 2, 3) == 3);
}
//==============================================================================
TEST_CASE("preprocessor_empty_variadic", "[preprocessor][empty_variadic]") {
  REQUIRE(TATOOINE_PP_EMPTY_VARIADIC() == 1);
  REQUIRE(TATOOINE_PP_EMPTY_VARIADIC(0, 2, 3) == 0);
}
//==============================================================================
TEST_CASE("preprocessor_invoke", "[preprocessor][invoke]") {
  auto f = [](auto x) { return x; };
  TATOOINE_PP_INVOKE(f);
  TATOOINE_PP_INVOKE(f, 1);
}
//==============================================================================
TEST_CASE("preprocessor_map", "[preprocessor][map]") {
  auto f = [](auto x) { return x; };
  auto g = [](auto x, auto y) { return x + y; };
  TATOOINE_PP_MAP(f)   // this should do nothing
  TATOOINE_PP_MAP2(f)  // this should do nothing
  REQUIRE(TATOOINE_PP_MAP(f, 1) == 1);
  REQUIRE(TATOOINE_PP_MAP2(g, 1, 2) == 3);
}
//==============================================================================
TEST_CASE("preprocessor_expand", "[preprocessor][expand]") {
  std::array arr{TATOOINE_PP_EXPAND((1))};
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
