#include <tatooine/einstein_notation.h>
#include <tatooine/tensor.h>
#include <tatooine/real.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::einstein_notation::test {
//==============================================================================
TEST_CASE("einstein_notation_indexed_tensors_to_indices",
          "[einstein_notation][indexed_tensors_to_indices]") {
  using indices =
      indexed_tensors_to_index_list<indexed_tensor<mat2, i_t, i_t>,
                                 j_t,
                                 indexed_tensor<vec2, k_t>>;
  REQUIRE(is_same<indices::at<0>, i_t>);
  REQUIRE(is_same<indices::at<1>, i_t>);
  REQUIRE(is_same<indices::at<2>, j_t>);
  REQUIRE(is_same<indices::at<3>, k_t>);
}
//==============================================================================
TEST_CASE("einstein_notation_free_indices",
          "[einstein_notation][free_indices]") {
  SECTION("just index list") {
    using indices = free_indices<i_t, i_t, j_t, k_t>;
    REQUIRE(indices::size == 2);
    REQUIRE(indices::contains<j_t>);
    REQUIRE(indices::contains<k_t>);
  }
  SECTION("indices from indexed tensor") {
    using indices = free_indices<indexed_tensor<mat2, i_t, i_t>,
                                 indexed_tensor<mat2, j_t, k_t>>;
    REQUIRE(indices::size == 2);
    REQUIRE(indices::contains<j_t>);
    REQUIRE(indices::contains<k_t>);
  }
}
//==============================================================================
TEST_CASE("einstein_notation_contracted_indices",
          "[einstein_notation][contracted_indices]") {
  SECTION("just index list") {
    using indices = contracted_indices<i_t, i_t, j_t, k_t>;
    REQUIRE(indices::size == 1);
    REQUIRE(indices::contains<i_t>);
  }
  SECTION("indices from indexed tensor") {
    using indices = contracted_indices<indexed_tensor<mat2, i_t, i_t>,
                                       indexed_tensor<mat2, j_t, k_t>>;
    REQUIRE(indices::size == 1);
    REQUIRE(indices::contains<i_t>);
  }
}
//==============================================================================
TEST_CASE("einstein_notation_index_map",
          "[einstein_notation][index_map]") {
  using Tjk = indexed_tensor<mat2, j_t, k_t>;
  auto map  = Tjk::index_map();

  REQUIRE(map[j_t::get()] == 0);
  REQUIRE(map[k_t::get()] == 1);
}
//==============================================================================
TEST_CASE("einstein_notation_contraction",
          "[einstein_notation][contraction]") {
  using Tij = indexed_tensor<mat2, i_t, j_t>;
  using Tjk = indexed_tensor<mat2, j_t, k_t>;
  using Tkl = indexed_tensor<mat2, k_t, l_t>;

  auto contracted_tensor = Tij{} * Tjk{} * Tkl{};

  using fi = decltype(contracted_tensor)::free_indices;
  using ci = decltype(contracted_tensor)::contracted_indices;

  REQUIRE(fi::size == 2);
  REQUIRE(fi::contains<i_t>);
  REQUIRE(fi::contains<l_t>);

  REQUIRE(ci::size == 2);
  REQUIRE(ci::contains<j_t>);
  REQUIRE(ci::contains<k_t>);
}
//==============================================================================
TEST_CASE("einstein_notation_contracted_assignement",
          "[einstein_notation][contracted_assignement]") {
  using Tij = indexed_tensor<mat2, i_t, j_t>;
  using Tjk = indexed_tensor<mat2, j_t, k_t>;
  using Tkl = indexed_tensor<mat2, k_t, l_t>;
  using Til = indexed_tensor<mat2, k_t, l_t>;

   Til{} = Tij{} * Tjk{} * Tkl{};
}
//==============================================================================
}  // namespace tatooine::einstein_notation::test
//==============================================================================
