#include <tatooine/symbolic.h>
#include <tatooine/tensor.h>
#include <tatooine/utility.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace Catch;
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("variadic_helpers", "[variadic_helpers]") {
  using namespace variadic;
  REQUIRE(front_number<1, 2, 3> == 1);
  REQUIRE(back_number<1, 2, 3> == 3);
  REQUIRE(ith_number<0, 1, 2, 3> == 1);
  REQUIRE(ith_number<1, 1, 2, 3> == 2);
  REQUIRE(ith_number<2, 1, 2, 3> == 3);

  SECTION("extract with equal types") {
    constexpr auto arr_0_4 = extract<0, 4>(1, 2, 3, 4);
    REQUIRE(size(arr_0_4) == 4);
    REQUIRE(arr_0_4[0] == 1);
    REQUIRE(arr_0_4[1] == 2);
    REQUIRE(arr_0_4[2] == 3);
    REQUIRE(arr_0_4[3] == 4);
    constexpr auto arr_1_4 = extract<1, 4>(1, 2, 3, 4);
    REQUIRE(size(arr_1_4) == 3);
    REQUIRE(arr_1_4[0] == 2);
    REQUIRE(arr_1_4[1] == 3);
    REQUIRE(arr_1_4[2] == 4);
    constexpr auto arr_0_2 = extract<0, 2>(1, 2, 3, 4);
    REQUIRE(size(arr_0_2) == 2);
    REQUIRE(arr_0_2[0] == 1);
    REQUIRE(arr_0_2[1] == 2);
    constexpr auto arr_1_3 = extract<1, 3>(1, 2, 3, 4);
    REQUIRE(size(arr_1_3) == 2);
    REQUIRE(arr_1_3[0] == 2);
    REQUIRE(arr_1_3[1] == 3);
    constexpr auto arr_2_4 = extract<2, 4>(1, 2, 3, 4);
    REQUIRE(size(arr_2_4) == 2);
    REQUIRE(arr_2_4[0] == 3);
    REQUIRE(arr_2_4[1] == 4);
  }

  SECTION("extract with diffent types") {
    {
      using tuple_type = extract_helper_tuple<0, 4, int, float, double, char>;
      REQUIRE(is_same<tuple_type::type_at<0>, int>);
      REQUIRE(is_same<tuple_type::type_at<1>, float>);
      REQUIRE(is_same<tuple_type::type_at<2>, double>);
      REQUIRE(is_same<tuple_type::type_at<3>, char>);
    }
    {
      using tuple_type = extract_helper_tuple<0, 2, int, double>;
      REQUIRE(is_same<tuple_type::type_at<0>, int>);
      REQUIRE(is_same<tuple_type::type_at<1>, double>);
    }
    {
      using tuple_type = extract_helper_tuple<1, 3, int, float, double, char>;
      REQUIRE(is_same<tuple_type::type_at<0>, float>);
      REQUIRE(is_same<tuple_type::type_at<1>, double>);
    }
    {
      using tuple_type = extract_helper_tuple<2, 4, int, float, double, char>;
      REQUIRE(is_same<tuple_type::type_at<0>, double>);
      REQUIRE(is_same<tuple_type::type_at<1>, char>);
    }

    auto extracted = extract<0, 2>(1, 2.0);
    REQUIRE(is_same<decltype(extracted)::type_at<0>, int>);
    REQUIRE(is_same<decltype(extracted)::type_at<1>, double>);
    REQUIRE(extracted.at<0>() == 1);
    REQUIRE(extracted.at<1>() == 2);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
