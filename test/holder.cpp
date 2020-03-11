#include <tatooine/holder.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("holder_lambda", "[holder][lambda]") {
  int i = 5;
  auto l = [&, j = 1]() mutable {
    return i++ + j++;
  };

  holder f0{l};
  holder f1{[&i] { ++i; return i; }};

  auto p0 = hold(l);
  auto p1 = hold([&, j = 1] { return i++ + j; });

  REQUIRE(std::is_reference_v<decltype(f0)::held_type>);
  REQUIRE(!std::is_reference_v<decltype(f1)::held_type>);
  REQUIRE(i == 5);
  REQUIRE(f0.get()() == 6);
  REQUIRE(i == 6);
  REQUIRE(f0.get()() == 8);
  REQUIRE(i == 7);
  f1.get()();
  //REQUIRE(i == 8);
  //REQUIRE(p1->evaluate() == 9);
  //REQUIRE(i == 9);
  //REQUIRE(p1->evaluate() == 10);
  //REQUIRE(i == 10);
  //REQUIRE(p0->evaluate() == 13);
  //REQUIRE(i == 11);
  //REQUIRE(p0->evaluate() == 15);
  //REQUIRE(i == 12);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
