#include <tatooine/unary_operation_field.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("unary_operation_field_identity", "[unary_operation_field][identity]"){
  analytical::fields::numerical::doublegyre v;
  auto v_id = make_unary_operation_field(
      v, [](auto const& x) -> auto const& { return x; },
      [](auto const& t) -> auto const& { return t; },
      [](auto const& ten) -> auto const& { return ten; });
  //using V = decltype(v);
  //using VId = decltype(v_id);
  //REQUIRE(V::num_dimensions() == VId::num_dimensions());
  //REQUIRE(std::is_same_v<V::real_t, VId::real_t>);
  //REQUIRE(std::is_same_v<V::tensor_t, VId::tensor_t>);
}
//==============================================================================
}
//==============================================================================
