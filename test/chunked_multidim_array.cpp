#include <tatooine/chunked_multidim_array.h>
#include <tatooine/random.h>
#include <tatooine/tensor.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("chunked_multidim_array_indices",
          "[chunked_multidim_array][indices]") {
  chunked_multidim_array<int> arr{{6, 4}, {3, 2}};
  arr.create_all_chunks();
  REQUIRE(arr.size(0) == 6);
  REQUIRE(arr.size(1) == 4);
  REQUIRE(arr.chunk_size(0) == 2);
  REQUIRE(arr.chunk_size(1) == 2);
  REQUIRE(arr.internal_chunk_size(0) == 3);
  REQUIRE(arr.internal_chunk_size(1) == 2);

  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 0) == 0);
  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 0) == 0);
  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 0) == 0);
  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 1) == 0);
  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 1) == 0);
  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 1) == 0);

  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 0) == 1);
  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 0) == 1);
  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 0) == 1);
  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 1) == 1);
  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 1) == 1);
  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 1) == 1);

  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 2) == 2);
  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 2) == 2);
  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 2) == 2);
  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 3) == 2);
  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 3) == 2);
  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 3) == 2);

  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 2) == 3);
  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 2) == 3);
  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 2) == 3);
  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 3) == 3);
  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 3) == 3);
  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 3) == 3);

  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 0, 0) == 0);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 1, 0) == 1);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 2, 0) == 2);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 0, 1) == 3);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 1, 1) == 4);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 2, 1) == 5);

  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 3, 0) == 0);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 4, 0) == 1);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 5, 0) == 2);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 3, 1) == 3);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 4, 1) == 4);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 5, 1) == 5);

  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 0, 2) == 0);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 1, 2) == 1);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 2, 2) == 2);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 0, 3) == 3);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 1, 3) == 4);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 2, 3) == 5);

  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 3, 2) == 0);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 4, 2) == 1);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 5, 2) == 2);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 3, 3) == 3);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 4, 3) == 4);
  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 5, 3) == 5);
}
//==============================================================================
TEMPLATE_TEST_CASE("chunked_multidim_array_element_assignment_primitive",
                   "[chunked_multidim_array]", int, float, double) {
  using T   = TestType;
  auto rand = []() {
    if constexpr (std::is_integral_v<T>) {
      return random_uniform<T>{0, 100}();
    } else {
      return random_uniform<T>{0.0, 1.0}();
    }
  };
  chunked_multidim_array<T> arr{{4, 4}, {2, 2}};
  for (size_t x = 0; x < 4; ++x) {
    for (size_t y = 0; y < 4; ++y) {
      auto v = rand();
      REQUIRE(arr(x, y) == T{});
      arr(x, y) = v;
      REQUIRE(arr(x, y) == v);
    }
  }
}
//==============================================================================
TEMPLATE_TEST_CASE("chunked_multidim_array_element_assignment_vec",
                   "[chunked_multidim_array][tensor]", float, double) {
  using T = vec<TestType, 2>;
  chunked_multidim_array<T> arr{{4, 4}, {2, 2}};
  for (size_t x = 0; x < 4; ++x) {
    for (size_t y = 0; y < 4; ++y) {
      auto v = T::randu();
      REQUIRE(approx_equal(arr(x, y), T{}));
      std::array is{x, y};
      arr(is) = v;
      REQUIRE(approx_equal(arr(x, y), v));
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
