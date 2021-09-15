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

  REQUIRE(map[0] == j_t::get());
  REQUIRE(map[1] == k_t::get());
}
//==============================================================================
TEST_CASE("einstein_notation_contraction",
          "[einstein_notation][contraction]") {
  mat2 A;
  auto Tij = indexed_tensor<mat2, i_t, j_t>{A};
  auto Tjk = indexed_tensor<mat2, j_t, k_t>{A};
  auto Tkl = indexed_tensor<mat2, k_t, l_t>{A};

  auto contracted_tensor = Tij * Tjk * Tkl;

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
  mat2 A;
  mat2 B{{1, 2},
         {3, 4}};
  mat2 C{{2, 3},
         {4, 5}};
  mat2 D{{3, 4},
         {5, 6}};
  SECTION("A(i, k) = B(i, j) * C(j, k) - standard matrix-matrix multiplication"){
    A(i, k)            = B(i, j) * C(j, k);
    auto const Amatrix = B * C;

    for_loop(
        [&](auto const... indices) {
          REQUIRE(A(indices...) == Approx(Amatrix(indices...)));
        },
        2, 2);
  }
  SECTION("A(i, k) = B(i, j) * C(k, j) - matrix- transposed matrix multiplication") {
    A(i, k)            = B(i, j) * C(k, j);
    auto const Amatrix = B * transposed(C);

    for_loop(
        [&](auto const... indices) {
          REQUIRE(A(indices...) == Approx(Amatrix(indices...)));
        },
        2, 2);
  }
  SECTION("A(i, l) = B(i, j) * C(j, k) * D(k, l) - matrix-matrix-matrix multiplication") {
    A(i, l)            = B(i, j) * C(j, k) * D(k, l);
    auto const Amatrix = B * C * D;

    for_loop(
        [&](auto const... indices) {
          REQUIRE(A(indices...) == Approx(Amatrix(indices...)));
        },
        2, 2);
  }
}
//==============================================================================
}  // namespace tatooine::einstein_notation::test
//==============================================================================
