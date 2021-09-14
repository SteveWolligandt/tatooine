#include <tatooine/type_list.h>
#include<catch2/catch.hpp>
//==============================================================================
namespace tatooine::test{
//==============================================================================
TEST_CASE("type_list", "[type_list]") {
  using l1 = type_list<int, int, float, double>;
  using l2 = type_list<float, int>;
  SECTION("size") {
    REQUIRE(l1::size == 4);
    REQUIRE(l2::size == 2);
  }
  SECTION("front") {
    REQUIRE(is_same<l1::front, int>);
    REQUIRE(is_same<l2::front, float>);
  }
  SECTION("back") {
    REQUIRE(is_same<l1::back, double>);
    REQUIRE(is_same<l2::back, int>);
  }
  SECTION("at") {
    REQUIRE(is_same<l1::at<0>, int>);
    REQUIRE(is_same<l1::at<1>, int>);
    REQUIRE(is_same<l1::at<2>, float>);
    REQUIRE(is_same<l1::at<3>, double>);
    REQUIRE(is_same<l2::at<0>, float>);
    REQUIRE(is_same<l2::at<1>, int>);
    //REQUIRE(is_same<type_list_at<l2, 2>,
    //                int>);  // This must yield a compiler-error.
  }
  SECTION("pop_back") {
    SECTION("l1") {
      using popped_back = l1::pop_back;
      REQUIRE(type_list_size<popped_back> == 3);
      REQUIRE(is_same<type_list_at<popped_back, 0>, int>);
      REQUIRE(is_same<type_list_at<popped_back, 1>, int>);
      REQUIRE(is_same<type_list_at<popped_back, 2>, float>);
    }
    SECTION("l2") {
      using popped_back = l2::pop_back;
      REQUIRE(type_list_size<popped_back> == 1);
      REQUIRE(is_same<type_list_at<popped_back, 0>, float>);
    }
  }
  SECTION("pop_front") {
    SECTION("l1") {
      using popped_front = l1::pop_front;
      REQUIRE(type_list_size<popped_front> == 3);
      REQUIRE(is_same<type_list_at<popped_front, 0>, int>);
      REQUIRE(is_same<type_list_at<popped_front, 1>, float>);
      REQUIRE(is_same<type_list_at<popped_front, 2>, double>);
    }
    SECTION("l2") {
      using popped_front = l2::pop_front;
      REQUIRE(type_list_size<popped_front> == 1);
      REQUIRE(is_same<type_list_at<popped_front, 0>, int>);
    }
  }
  SECTION("push_back") {
    SECTION("l1") {
      using pushed_back = l1::push_back<char>;
      REQUIRE(type_list_size<pushed_back> == 5);
      REQUIRE(is_same<type_list_at<pushed_back, 0>, int>);
      REQUIRE(is_same<type_list_at<pushed_back, 1>, int>);
      REQUIRE(is_same<type_list_at<pushed_back, 2>, float>);
      REQUIRE(is_same<type_list_at<pushed_back, 3>, double>);
      REQUIRE(is_same<type_list_at<pushed_back, 4>, char>);
    }
    SECTION("l2") {
      using pushed_back = l2::push_back<char>;
      REQUIRE(type_list_size<pushed_back> == 3);
      REQUIRE(is_same<type_list_at<pushed_back, 0>, float>);
      REQUIRE(is_same<type_list_at<pushed_back, 1>, int>);
      REQUIRE(is_same<type_list_at<pushed_back, 2>, char>);
    }
  }
  SECTION("push_front") {
    SECTION("l1") {
      using pushed_front = l1::push_front<char>;
      REQUIRE(type_list_size<pushed_front> == 5);
      REQUIRE(is_same<type_list_at<pushed_front, 0>, char>);
      REQUIRE(is_same<type_list_at<pushed_front, 1>, int>);
      REQUIRE(is_same<type_list_at<pushed_front, 2>, int>);
      REQUIRE(is_same<type_list_at<pushed_front, 3>, float>);
      REQUIRE(is_same<type_list_at<pushed_front, 4>, double>);
    }
    SECTION("l2") {
      using pushed_front = l2::push_front<char>;
      REQUIRE(type_list_size<pushed_front> == 3);
      REQUIRE(is_same<type_list_at<pushed_front, 0>, char>);
      REQUIRE(is_same<type_list_at<pushed_front, 1>, float>);
      REQUIRE(is_same<type_list_at<pushed_front, 2>, int>);
    }
  }
  SECTION("contains") {
    REQUIRE(l1::contains<int>);
    REQUIRE(l1::contains<float>);
    REQUIRE(l1::contains<double>);
    REQUIRE_FALSE(l1::contains<char>);
    REQUIRE(l2::contains<int>);
    REQUIRE(l2::contains<float>);
    REQUIRE_FALSE(l2::contains<double>);
    REQUIRE_FALSE(l2::contains<char>);
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
