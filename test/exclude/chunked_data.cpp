#include <tatooine/chunked_data.h>
#include <tatooine/random.h>
#include <tatooine/tensor.h>
#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEMPLATE_TEST_CASE("chunked_data_element_assignment_primitive",
                   "[chunked_data]", int, float, double) {
  using T = TestType;
  auto rand = []() {
    if constexpr (std::is_integral_v<T>) {
      return random_uniform<T>{0, 100}();
    } else {
      return random_uniform<T>{0.0, 1.0}();
    }
  };
  chunked_data<T, 2, 2> data{4, 4};
  for (size_t x = 0; x < 4; ++x) {
    for (size_t y = 0; y < 4; ++y) {
      auto v = rand();
      REQUIRE(data(x, y) == T{});
      data(x, y) = v;
      REQUIRE(data(x, y) == v);
    }
  }
}
//==============================================================================
TEMPLATE_TEST_CASE("chunked_data_element_assignment_vec",
                   "[chunked_data][tensor]", float, double) {
  using T = vec<TestType, 2>;
  chunked_data<T, 2, 2> data{4, 4};
  for (size_t x = 0; x < 4; ++x) {
    for (size_t y = 0; y < 4; ++y) {
      auto v = T::randu();
      REQUIRE(approx_equal(data(x, y), T{}));
      std::array is{x, y};
      data(is) = v;
      REQUIRE(approx_equal(data(x, y), v));
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
