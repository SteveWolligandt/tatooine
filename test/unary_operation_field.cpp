#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/unary_operation_field.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("unary_operation_field_identity_move",
          "[unary_operation_field][identity][move]") {
  constexpr auto identity = [](auto&& p) -> decltype(auto) {
    return std::forward<decltype(p)>(p);
  };
  auto v  = analytical::numerical::doublegyre{} | identity;
  using V = decltype(v);
  REQUIRE(!std::is_reference_v<V::internal_field_t>);
  REQUIRE(std::is_same_v<V::internal_field_t,
                         analytical::numerical::doublegyre<double>>);
}
//==============================================================================
TEST_CASE("unary_operation_field_identity_ref",
          "[unary_operation_field][identity][ref]") {
  analytical::numerical::doublegyre v;
  constexpr auto                    identity = [](auto&& p) -> decltype(auto) {
    return std::forward<decltype(p)>(p);
  };
  auto v_id = v | identity;
  using V   = decltype(v);
  using VId = decltype(v_id);
  REQUIRE(std::is_same_v<V::real_type, VId::real_type>);
  REQUIRE(std::is_same_v<V::tensor_type, VId::tensor_type>);
  for (auto t : linspace(0.0, 10.0, 10)) {
    for (auto y : linspace(0.0, 1.0, 10)) {
      for (auto x : linspace(0.0, 2.0, 20)) {
        vec pos{x, y};
        REQUIRE(v(pos, t) == v_id(pos, t));
      }
    }
  }
}
//==============================================================================
TEST_CASE("unary_operation_field_identity_ptr",
          "[unary_operation_field][identity][ptr][pointer]") {
  analytical::numerical::doublegyre    v;
  polymorphic::vectorfield<double, 2>* v_ptr = &v;
  constexpr auto identity                    = [](auto&& p) -> decltype(auto) {
    return std::forward<decltype(p)>(p);
  };
  auto v_id = v_ptr | identity;
  using V   = decltype(v);
  using VId = decltype(v_id);
  REQUIRE(v_ptr == v_id.internal_field());
  REQUIRE(std::is_same_v<V::real_type, VId::real_type>);
  REQUIRE(std::is_same_v<V::tensor_type, VId::tensor_type>);
  for (auto t : linspace(0.0, 10.0, 10)) {
    for (auto y : linspace(0.0, 1.0, 10)) {
      for (auto x : linspace(0.0, 2.0, 20)) {
        vec pos{x, y};
        REQUIRE(v(pos, t) == v_id(pos, t));
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
  REQUIRE(std::is_same_v<V::real_type, VLen::real_type>);
  REQUIRE(std::is_same_v<VLen::tensor_type, double>);
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
  using VDLenInt    = typename VDLen::internal_field_t;
  using VDLenIntInt = typename VDLenInt::internal_field_t;
  REQUIRE(!std::is_reference_v<VDLenInt>);
  REQUIRE(std::is_reference_v<VDLenIntInt>);
  REQUIRE(std::is_same_v<std::decay_t<VDLenIntInt>, V>);

  REQUIRE(std::is_same_v<V::real_type, VDLen::real_type>);
  REQUIRE(std::is_same_v<VDLen::tensor_type, double>);
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
