#include <tatooine/chunked_multidim_array.h>
#include <tatooine/random.h>
#include <tatooine/tensor.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("chunked_multidim_array_iterating",
          "[chunked_multidim_array][iterating]") {
  auto const arr =
      chunked_multidim_array<int>{std::array{5, 4}, std::array{2, 2}};
  std::vector<std::vector<size_t>> indices;
  arr.iterate_over_indices([&indices](auto const& is){indices.push_back(is);});

  // first chunk
  REQUIRE(indices[0][0] == 0); REQUIRE(indices[0][1] == 0);
  REQUIRE(indices[1][0] == 1); REQUIRE(indices[1][1] == 0);
  REQUIRE(indices[2][0] == 0); REQUIRE(indices[2][1] == 1);
  REQUIRE(indices[3][0] == 1); REQUIRE(indices[3][1] == 1);
  // second chunk
  REQUIRE(indices[4][0] == 2); REQUIRE(indices[4][1] == 0);
  REQUIRE(indices[5][0] == 3); REQUIRE(indices[5][1] == 0);
  REQUIRE(indices[6][0] == 2); REQUIRE(indices[6][1] == 1);
  REQUIRE(indices[7][0] == 3); REQUIRE(indices[7][1] == 1);
  // third chunk
  REQUIRE(indices[8][0] == 4); REQUIRE(indices[8][1] == 0);
  REQUIRE(indices[9][0] == 4); REQUIRE(indices[9][1] == 1);
  // forth chunk
  REQUIRE(indices[10][0] == 0); REQUIRE(indices[10][1] == 2);
  REQUIRE(indices[11][0] == 1); REQUIRE(indices[11][1] == 2);
  REQUIRE(indices[12][0] == 0); REQUIRE(indices[12][1] == 3);
  REQUIRE(indices[13][0] == 1); REQUIRE(indices[13][1] == 3);
  // fifth chunk
  REQUIRE(indices[14][0] == 2); REQUIRE(indices[14][1] == 2);
  REQUIRE(indices[15][0] == 3); REQUIRE(indices[15][1] == 2);
  REQUIRE(indices[16][0] == 2); REQUIRE(indices[16][1] == 3);
  REQUIRE(indices[17][0] == 3); REQUIRE(indices[17][1] == 3);
  // sixth chunk
  REQUIRE(indices[18][0] == 4); REQUIRE(indices[18][1] == 2);
  REQUIRE(indices[19][0] == 4); REQUIRE(indices[19][1] == 3);
}
//==============================================================================
TEST_CASE("chunked_multidim_array_regular_and_irregular_chunks",
          "[chunked_multidim_array][regular][irregular]") {
  chunked_multidim_array<int> arr{std::array{10, 8}, std::array{3, 2}};
  arr.create_all_chunks();
  auto const& chunk_0_0 = arr.chunk_at(0, 0);
  auto const& chunk_3_0 = arr.chunk_at(3, 0);
  auto const& chunk_3_3 = arr.chunk_at(3, 3);
  auto const& chunk_0_3 = arr.chunk_at(0, 3);
  REQUIRE(chunk_0_0->size()[0] == 3);
  REQUIRE(chunk_0_0->size()[1] == 2);
  REQUIRE(chunk_3_0->size()[0] == 1);
  REQUIRE(chunk_3_0->size()[1] == 2);
  REQUIRE(chunk_3_3->size()[0] == 1);
  REQUIRE(chunk_3_3->size()[1] == 2);
  REQUIRE(chunk_0_3->size()[0] == 3);
  REQUIRE(chunk_0_3->size()[1] == 2);
}
//==============================================================================
TEST_CASE("chunked_multidim_array_regular_chunks",
          "[chunked_multidim_array][regular]") {
  chunked_multidim_array<int> arr{std::array{8, 8}, std::array{2, 2}};
  arr.create_all_chunks();
  auto const& chunk_0_0 = arr.chunk_at(0, 0);
  auto const& chunk_3_0 = arr.chunk_at(3, 0);
  auto const& chunk_3_3 = arr.chunk_at(3, 3);
  auto const& chunk_0_3 = arr.chunk_at(0, 3);
  REQUIRE(chunk_0_0->size()[0] == 2);
  REQUIRE(chunk_0_0->size()[1] == 2);
  REQUIRE(chunk_3_0->size()[0] == 2);
  REQUIRE(chunk_3_0->size()[1] == 2);
  REQUIRE(chunk_3_3->size()[0] == 2);
  REQUIRE(chunk_3_3->size()[1] == 2);
  REQUIRE(chunk_0_3->size()[0] == 2);
  REQUIRE(chunk_0_3->size()[1] == 2);
}
//==============================================================================
TEST_CASE("chunked_multidim_array_irregular_chunks",
          "[chunked_multidim_array][irregular]") {
  chunked_multidim_array<int> arr{std::array{10, 10}, std::array{3, 3}};
  arr.create_all_chunks();
  auto const& chunk_0_0 = arr.chunk_at(0, 0);
  auto const& chunk_3_0 = arr.chunk_at(3, 0);
  auto const& chunk_3_3 = arr.chunk_at(3, 3);
  auto const& chunk_0_3 = arr.chunk_at(0, 3);
  REQUIRE(chunk_0_0->size()[0] == 3);
  REQUIRE(chunk_0_0->size()[1] == 3);
  REQUIRE(chunk_3_0->size()[0] == 1);
  REQUIRE(chunk_3_0->size()[1] == 3);
  REQUIRE(chunk_3_3->size()[0] == 1);
  REQUIRE(chunk_3_3->size()[1] == 1);
  REQUIRE(chunk_0_3->size()[0] == 3);
  REQUIRE(chunk_0_3->size()[1] == 1);
}
//==============================================================================
TEST_CASE("chunked_multidim_array_indices",
          "[chunked_multidim_array][indices]") {
  SECTION("x_fastest") {
    chunked_multidim_array<int, x_fastest> arr{std::array{6, 4},
                                               std::array{3, 2}};
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
  //SECTION("x_slowest") {
  //  chunked_multidim_array<int, x_slowest> arr{std::array{6, 4},
  //                                             std::array{3, 2}};
  //  arr.create_all_chunks();
  //  REQUIRE(arr.size(0) == 6);
  //  REQUIRE(arr.size(1) == 4);
  //  REQUIRE(arr.chunk_size(0) == 2);
  //  REQUIRE(arr.chunk_size(1) == 2);
  //  REQUIRE(arr.internal_chunk_size(0) == 3);
  //  REQUIRE(arr.internal_chunk_size(1) == 2);
  //
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 0) == 0);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 0) == 0);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 0) == 0);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 1) == 0);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 1) == 0);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 1) == 0);
  //
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 2) == 1);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 2) == 1);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 2) == 1);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(0, 3) == 1);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(1, 3) == 1);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(2, 3) == 1);
  //
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 0) == 2);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 0) == 2);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 0) == 2);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 1) == 2);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 1) == 2);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 1) == 2);
  //
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 2) == 3);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 2) == 3);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 2) == 3);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(3, 3) == 3);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(4, 3) == 3);
  //  REQUIRE(arr.plain_chunk_index_from_global_indices(5, 3) == 3);
  //
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 0, 0) == 0);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 0, 1) == 1);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 1, 0) == 2);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 1, 1) == 3);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 2, 0) == 4);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(0, 2, 1) == 5);
  //
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 3, 0) == 0);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 3, 1) == 1);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 4, 0) == 2);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 4, 1) == 3);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 5, 0) == 4);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(1, 5, 1) == 5);
  //
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 0, 2) == 0);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 0, 3) == 1);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 1, 2) == 2);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 1, 3) == 3);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 2, 2) == 4);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(2, 2, 3) == 5);
  //
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 3, 2) == 0);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 3, 3) == 1);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 4, 2) == 2);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 4, 3) == 3);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 5, 2) == 4);
  //  REQUIRE(arr.plain_internal_chunk_index_from_global_indices(3, 5, 3) == 5);
  //}
}
//==============================================================================
TEMPLATE_TEST_CASE("chunked_multidim_array_element_assignment_primitive",
                   "[chunked_multidim_array]", int, float, double) {
  using T   = TestType;
  auto rand = []() {
    if constexpr (std::is_integral_v<T>) {
      return random::uniform<T>{0, 100}();
    } else {
      return random::uniform<T>{0.0, 1.0}();
    }
  };
  chunked_multidim_array<T> arr{std::vector<size_t>{4, 4}, std::vector<size_t>{2, 2}};
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
  chunked_multidim_array<T> arr{std::vector<size_t>{5, 4},
                                std::vector<size_t>{2, 2}};
  for (size_t x = 0; x < 5; ++x) {
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
