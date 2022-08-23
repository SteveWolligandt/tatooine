#include <tatooine/type_traits.h>
#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("type_traits_sfinae", "[type_traits][sfinae]") {
  struct non_const_method {
    void foo() {}
  };
  struct const_method {
    void foo() const {}
  };
  struct both_methods {
    int i {};
    int& foo() {return i;}
    int foo() const {return i;}
  };

  REQUIRE(std::is_reference_v<decltype(std::declval<both_methods>().foo())>);
  REQUIRE(
      !std::is_reference_v<decltype(std::declval<both_methods const>().foo())>);

  REQUIRE(std::is_invocable_v<decltype(&const_method::foo), const_method &>);
  REQUIRE(std::is_invocable_v<decltype(&const_method::foo), const_method const&>);
  REQUIRE(std::is_invocable_v<decltype(&non_const_method::foo), non_const_method &>);
  REQUIRE_FALSE(std::is_invocable_v<decltype(&non_const_method::foo), non_const_method const&>);
  REQUIRE_FALSE(
      std::is_invocable_v<decltype(&const_method::foo), const_method &, int>);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
