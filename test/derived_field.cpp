#include "../derived_field.h"
#include "../doublegyre.h"
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("derived_field_numerical_doublegyre", "[derived_field]") {
  numerical::doublegyre dg;
  auto                  ddg  = diff(dg);
  REQUIRE(decltype(ddg)::tensor_t::num_dimensions() == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(1) == 2);
  std::cerr << ddg({0.5, 0.5}, 0) << '\n';

  auto                  dddg = diff(ddg);
  REQUIRE(decltype(dddg)::tensor_t::num_dimensions() == 3);
  REQUIRE(decltype(dddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(1) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(2) == 2);
  std::cerr << dddg({0.5, 0.5}, 0).slice<2>(0) << '\n';
  std::cerr << dddg({0.5, 0.5}, 0).slice<2>(1) << '\n';
}

TEST_CASE("derived_field_symbolic_doublegyre", "[derived_field]") {
  symbolic::doublegyre  dg;
  [[maybe_unused]] auto& dg_expr = dg.expr();
  std::cerr << dg_expr << '\n';

  auto                   ddg      = diff(dg);
  [[maybe_unused]] auto& ddg_expr = ddg.expr();
  std::cerr << ddg_expr << '\n';
  REQUIRE(decltype(ddg)::tensor_t::num_dimensions() == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(1) == 2);
  std::cerr << ddg({0.5, 0.5}, 0) << '\n';

  auto                  dddg      = diff(ddg);
  [[maybe_unused]] auto& dddg_expr = dddg.expr();
  std::cerr << dddg_expr.slice<2>(0) << '\n';
  std::cerr << dddg_expr.slice<2>(1) << '\n';
  REQUIRE(decltype(dddg)::tensor_t::num_dimensions() == 3);
  REQUIRE(decltype(dddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(1) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(2) == 2);
  std::cerr << dddg({0.5, 0.5}, 0).slice<2>(0) << '\n';
  std::cerr << dddg({0.5, 0.5}, 0).slice<2>(1) << '\n';
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
