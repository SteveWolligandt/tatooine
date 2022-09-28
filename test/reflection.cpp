#include <tatooine/reflection.h>
#include <tatooine/concepts.h>

#include <catch2/catch_test_macros.hpp>
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
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
namespace tatooine::reflection {
TATOOINE_MAKE_ADT_REFLECTABLE(tatooine::test::reflection_test,
                              TATOOINE_REFLECTION_INSERT_GETTER(i),
                              TATOOINE_REFLECTION_INSERT_GETTER(j));
}
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("reflection", "[reflection]") {
  auto obj = reflection_test{-1, 2};
  reflection::for_each(obj, [](auto name, auto&& val) {
    using value_type = decltype(val);
    if (name == "i") {
      REQUIRE(same_as<value_type, int&>);
      if constexpr (same_as<value_type, int&>) {
        REQUIRE(val == -1);
      }
    } else if (name == "j") {
      REQUIRE(same_as<value_type, float&>);
      if constexpr (same_as<value_type, float&>) {
        REQUIRE(val == 2.0f);
      }
    }
  });
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
