#include <tatooine/newton_raphson.h>
#include <tatooine/doublegyre.h>
#include <tatooine/sinuscosinus.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

struct xy_field : symbolic::field<double, 2, 2> {
  using this_t   = xy_field;
  using parent_t = symbolic::field<double, 2, 2>;
  using parent_t::t;
  using parent_t::x;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  xy_field() { this->set_expr(vec{GiNaC::ex{x(0)}, GiNaC::ex{x(1)}}); }
};

//==============================================================================
template <typename V>
auto newton_raphson_test(V& v, const typename V::pos_t& x,
                         const typename V::real_t t,
                         const typename V::pos_t& expected) {
  auto x0 = newton_raphson(v, x, t, 1000);
  INFO("x0 = " << x0);
  REQUIRE(approx_equal(x0, expected));
}

//==============================================================================
TEST_CASE("newton_raphson_xy", "[newton_raphson][xy][symbolic]") {
  xy_field xy;
  newton_raphson_test(xy, {1,1}, 0.0, {0,0});
}

//==============================================================================
TEST_CASE("newton_raphson_doublegyre", "[newton_raphson][doublegyre][dg][symbolic]") {
  symbolic::doublegyre dg;
  newton_raphson_test(dg, {1.1, 0.1}, 0, {1, 0});
  newton_raphson_test(dg, {0.9, 0.1}, 0, {1, 0});
  newton_raphson_test(dg, {0.9, -0.1}, 0, {1, 0});
  newton_raphson_test(dg, {1.1, -0.1}, 0, {1, 0});
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
