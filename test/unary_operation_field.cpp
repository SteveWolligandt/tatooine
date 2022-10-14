#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/unary_operation_field.h>
#include <tatooine/test/ApproxRange.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("unary_operation_field_identity_rvalue",
          "[unary_operation_field][identity][rvalue]") {
  constexpr auto identity = [](auto&& p) -> decltype(auto) {
    return std::forward<decltype(p)>(p);
  };
  auto v  = analytical::numerical::doublegyre{} | identity;
  using V = decltype(v);
  REQUIRE(!std::is_reference_v<V::internal_field_type>);
  REQUIRE(is_same<V::internal_field_type,
                  analytical::numerical::doublegyre<double>>);
}
//==============================================================================
TEST_CASE("unary_operation_field_identity_ref",
          "[unary_operation_field][identity][ref]") {
  auto           v        = analytical::numerical::doublegyre{};
  constexpr auto identity = [](auto&& p) -> decltype(auto) {
    return std::forward<decltype(p)>(p);
  };
  auto       v_id = v | identity;
  using V         = decltype(v);
  using VId       = decltype(v_id);
  CHECK(is_same<field_real_type<V>, field_real_type<VId>>);

  {
    CAPTURE(type_name<V::tensor_type>(), type_name<VId::tensor_type>());
    CHECK(is_same<field_tensor_type<V>, field_tensor_type<VId>>);
  }

  for (auto t : linspace(0.0, 10.0, 10)) {
    for (auto y : linspace(0.0, 1.0, 10)) {
      for (auto x : linspace(0.0, 2.0, 20)) {
        auto const pos = vec{x, y};
        CAPTURE(pos, t);
        REQUIRE_THAT(v(pos, t), ApproxRange(v_id(pos, t)).margin(1e-10));
      }
    }
  }
}
//==============================================================================
TEST_CASE("unary_operation_field_identity_ptr",
          "[unary_operation_field][identity][ptr][pointer]") {
  analytical::numerical::doublegyre         v;
  polymorphic::vectorfield<real_number, 2>* v_ptr = &v;
  constexpr auto identity = [](auto&& p) -> decltype(auto) {
    return std::forward<decltype(p)>(p);
  };
  auto v_id = v_ptr | identity;
  using V   = decltype(v);
  using VId = decltype(v_id);
  REQUIRE(v_ptr == v_id.internal_field());
  REQUIRE(is_same<V::real_type, VId::real_type>);

  {
    CAPTURE(type_name<V::tensor_type>(), type_name<VId::tensor_type>());
    CHECK(is_same<V::tensor_type, VId::tensor_type>);
  }

  for (auto t : linspace(0.0, 10.0, 10)) {
    for (auto y : linspace(0.0, 1.0, 10)) {
      for (auto x : linspace(0.0, 2.0, 20)) {
        auto const pos = vec{x, y};
        REQUIRE_THAT(v(pos, t), ApproxRange(v_id(pos, t)).margin(1e-10));
      }
    }
  }
}
//==============================================================================
TEST_CASE("unary_operation_field_length", "[unary_operation_field][length]") {
  auto v     = analytical::numerical::doublegyre{};
  auto v_len = v | [](auto const& v) { return euclidean_length(v); };
  using V    = decltype(v);
  using VLen = decltype(v_len);
  REQUIRE(is_same<V::real_type, VLen::real_type>);
  REQUIRE(is_same<VLen::tensor_type, real_number>);
  for (auto t : linspace{0.0, 10.0, 10}) {
    for (auto y : linspace{0.0, 1.0, 10}) {
      for (auto x : linspace{0.0, 2.0, 20}) {
        REQUIRE(euclidean_length(v(x, y, t)) == v_len(x, y, t));
      }
    }
  }
}
//==============================================================================
TEST_CASE("unary_operation_field_concat", "[unary_operation_field][concat]") {
  analytical::numerical::doublegyre v;
  auto v_double_len = v | [](auto const& v) { return euclidean_length(v); } |
                      [](auto const l) { return l * 2; };

  using V           = decltype(v);
  using VDLen       = decltype(v_double_len);
  using VDLenInt    = typename VDLen::internal_field_type;
  using VDLenIntInt = typename VDLenInt::internal_field_type;
  REQUIRE(!std::is_reference_v<VDLenInt>);
  REQUIRE(std::is_reference_v<VDLenIntInt>);
  REQUIRE(is_same<std::decay_t<VDLenIntInt>, V>);

  REQUIRE(is_same<V::real_type, VDLen::real_type>);
  REQUIRE(is_same<VDLen::tensor_type, real_number>);
  for (auto t : linspace(0.0, 10.0, 10)) {
    for (auto y : linspace(0.0, 1.0, 10)) {
      for (auto x : linspace(0.0, 2.0, 20)) {
        REQUIRE(euclidean_length(v(x, y, t)) * 2 == v_double_len(x, y, t));
      }
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
