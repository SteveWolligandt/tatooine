#include "../doublegyre.h"
#include "../derived_field.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("symbolic_field_acceleration", "[symbolic_field]") {
  symbolic::doublegyre dg;
  auto jdg = diff(dg);
  auto dg_acc = jdg * dg;
  auto jdg_acc = diff(dg_acc);
  auto dg_jerk = jdg_acc * dg;

  std::cerr << dg_acc.expr() << '\n'<< '\n';
  std::cerr << dg_jerk.expr() << '\n'<< '\n';
  std::cout << dg_acc({0.3, 0.3}, 0) << '\n';
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
