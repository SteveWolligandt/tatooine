#include <tatooine/reflection.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
struct reflection_test {
 private:
  int   m_i;
  float m_j;

 public:
  reflection_test(int i, float j) : m_i{i}, m_j{j} {}
  auto i() -> auto& {
    return m_i;
  }
  auto i() const {
    return m_i;
  }
  auto j() -> auto& {
    return m_j;
  }
  auto j() const {
    return m_j;
  }
};
}  // namespace tatooine::test
TATOOINE_MAKE_ADT_REFLECTABLE(tatooine::test::reflection_test,
                              TATOOINE_REFLECTION_INSERT_GETTER(i),
                              TATOOINE_REFLECTION_INSERT_GETTER(j));
namespace tatooine::test {
TEST_CASE("reflection", "[reflection]") {
  reflection_test obj{1, 2};
  reflection::for_each(obj, [](auto name, auto&& val) {
    if (name == "i") {
      REQUIRE(val == 1);
      REQUIRE(std::is_same_v<decltype(val), int&>);
    } else if (name == "j") {
      REQUIRE(val == 2);
      REQUIRE(std::is_same_v<decltype(val), float&>);
    }
  });
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
