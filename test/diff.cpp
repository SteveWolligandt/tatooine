#include <tatooine/diff.h>
#include <tatooine/doublegyre.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("derived_field_numerical_doublegyre",
          "[derived_field][numerical][doublegyre]") {
  numerical::doublegyre dg;
  auto                  ddg  = diff(dg);
  REQUIRE(decltype(ddg)::tensor_t::num_dimensions() == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(1) == 2);

  auto                  dddg = diff(ddg);
  REQUIRE(decltype(dddg)::tensor_t::num_dimensions() == 3);
  REQUIRE(decltype(dddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(1) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(2) == 2);
}

//==============================================================================
TEST_CASE("derived_field_symbolic_doublegyre",
          "[derived_field][symbolic][doublegyre]") {
  symbolic::doublegyre  dg;
  [[maybe_unused]] auto& dg_expr = dg.expr();

  auto                   ddg      = diff(dg);
  [[maybe_unused]] auto& ddg_expr = ddg.expr();
  REQUIRE(decltype(ddg)::tensor_t::num_dimensions() == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(ddg)::tensor_t::dimension(1) == 2);

  auto                  dddg      = diff(ddg);
  [[maybe_unused]] auto& dddg_expr = dddg.expr();
  REQUIRE(decltype(dddg)::tensor_t::num_dimensions() == 3);
  REQUIRE(decltype(dddg)::tensor_t::dimension(0) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(1) == 2);
  REQUIRE(decltype(dddg)::tensor_t::dimension(2) == 2);
}

//==============================================================================
TEST_CASE("derived_field_linear_field", "[derived_field][symbolic]") {
  auto& x0 = symbolic::symbol::x(0);
  auto& x1 = symbolic::symbol::x(1);
  symbolic::field<double, 2, 2> f{
      {x0 * x1 * 2,
       x1 * 3}};
  auto df = diff(f);
  REQUIRE(df.expr()(0,0) == GiNaC::diff(f.expr()(0), x0));
  REQUIRE(df.expr()(1,0) == GiNaC::diff(f.expr()(1), x0));
  REQUIRE(df.expr()(0,1) == GiNaC::diff(f.expr()(0), x1));
  REQUIRE(df.expr()(1,1) == GiNaC::diff(f.expr()(1), x1));
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
