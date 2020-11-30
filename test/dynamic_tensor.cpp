#include <tatooine/dynamic_tensor.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("dynamic_tensor_transposed", "[dynamic][tensor][transpose][transposed]"){
  auto A = dynamic_tensor<double>::ones(3, 3);
  A(1,0) = 3;
  A(2,1) = 5;
  SECTION("non-const") {
    auto At = transposed(A);
    STATIC_REQUIRE(
        std::is_same_v<decltype(At),
                       transposed_dynamic_tensor<dynamic_tensor<double>>>);
    REQUIRE(At(0, 1) == 3);
    REQUIRE(At(1, 2) == 5);
  }
  SECTION("const") {
    auto const B  = A;
    auto       Bt = transposed(B);
    STATIC_REQUIRE(std::is_same_v<decltype(Bt), const_transposed_dynamic_tensor<
                                                    dynamic_tensor<double>>>);
    REQUIRE(Bt(0, 1) == 3);
    REQUIRE(Bt(1, 2) == 5);
  }
}
//==============================================================================
TEST_CASE("dynamic_tensor_diag", "[dynamic][tensor][diag]"){
  auto a = dynamic_tensor<double>::ones(3);
  a(1) = 2;
  a(2) = 3;
  SECTION("non-const") {
    auto A = diag(a);
    STATIC_REQUIRE(
        std::is_same_v<decltype(A), diag_dynamic_tensor<decltype(a)>>);
    REQUIRE(A(0, 0) == 1);
    REQUIRE(A(1, 0) == 0);
    REQUIRE(A(2, 0) == 0);
    REQUIRE(A(0, 1) == 0);
    REQUIRE(A(1, 1) == 2);
    REQUIRE(A(2, 1) == 0);
    REQUIRE(A(0, 2) == 0);
    REQUIRE(A(1, 2) == 0);
    REQUIRE(A(2, 2) == 3);
  }
  SECTION("const"){
    auto const b = a;
    auto       B = diag(b);
    STATIC_REQUIRE(
        std::is_same_v<decltype(B),
                       const_diag_dynamic_tensor<dynamic_tensor<double>>>);
    REQUIRE(B(0, 0) == 1);
    REQUIRE(B(1, 0) == 0);
    REQUIRE(B(2, 0) == 0);
    REQUIRE(B(0, 1) == 0);
    REQUIRE(B(1, 1) == 2);
    REQUIRE(B(2, 1) == 0);
    REQUIRE(B(0, 2) == 0);
    REQUIRE(B(1, 2) == 0);
    REQUIRE(B(2, 2) == 3);
  }
}
//==============================================================================
TEST_CASE("dynamic_tensor_contraction", "[dynamic][tensor][contraction]") {
  SECTION("matrix-vector-multiplication") {
    dynamic_tensor<double> A{3, 3}, b{3}, c{3};

    A(0, 0) = 2.37546;
    A(0, 1) = 8.93258;
    A(0, 2) = 7.19590;
    A(1, 0) = 0.12527;
    A(1, 1) = 5.39804;
    A(1, 2) = 4.91963;
    A(2, 0) = 2.62273;
    A(2, 1) = 0.25176;
    A(2, 2) = 2.42668;
    b(0)    = 1.6209;
    b(1)    = 5.2425;
    b(2)    = 9.6812;
    c(0)    = 120.344;
    c(1)    = 76.130;
    c(2)    = 29.064;
    auto const d = A * b;
    REQUIRE(d(0) == Approx(c(0)));
    REQUIRE(d(1) == Approx(c(1)));
    REQUIRE(d(2) == Approx(c(2)));
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
