#include <tatooine/einstein_notation.h>
#include <tatooine/real.h>
#include <tatooine/tensor.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::einstein_notation::test {
//==============================================================================
TEST_CASE("einstein_notation_indexed_tensors_to_indices",
          "[einstein_notation][indexed_tensors_to_indices]") {
  using indices = indexed_tensors_to_index_list<indexed_tensor<mat2, i_t, i_t>,
                                                j_t, indexed_tensor<vec2, k_t>>;
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
TEST_CASE("einstein_notation_index_map", "[einstein_notation][index_map]") {
  using Tjk = indexed_tensor<mat2, j_t, k_t>;
  auto map  = Tjk::index_map();

  REQUIRE(map[0] == j_t::get());
  REQUIRE(map[1] == k_t::get());
}
//==============================================================================
TEST_CASE("einstein_notation_contraction", "[einstein_notation][contraction]") {
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
  mat2 B{{1, 2}, {3, 4}};
  mat2 C{{2, 3}, {4, 5}};
  mat2 D{{3, 4}, {5, 6}};
  SECTION(
      "A(i, k) = B(i, j) * C(j, k) - standard matrix-matrix multiplication") {
    A(i, k)            = B(i, j) * C(j, k);
    auto const Amatrix = B * C;

    for_loop(
        [&](auto const... is) { REQUIRE(A(is...) == Approx(Amatrix(is...))); },
        2, 2);
  }
  SECTION(
      "A(i, k) = B(i, j) * C(k, j) - matrix- transposed matrix "
      "multiplication") {
    A(i, k)            = B(i, j) * C(k, j);
    auto const Amatrix = B * transposed(C);

    for_loop(
        [&](auto const... is) { REQUIRE(A(is...) == Approx(Amatrix(is...))); },
        2, 2);
  }
  SECTION(
      "A(i, l) = B(i, j) * C(j, k) * D(k, l) - matrix-matrix-matrix "
      "multiplication") {
    A(i, l)            = B(i, j) * C(j, k) * D(k, l);
    auto const Amatrix = B * C * D;

    for_loop(
        [&](auto const... is) { REQUIRE(A(is...) == Approx(Amatrix(is...))); },
        2, 2);
  }
  SECTION("A(i, j) = B(i, j) + C(i, j) - matrix-matrix addition") {
    A(i, j) = B(i, j) + C(i, j);

    for_loop(
        [&](auto const... is) {
          REQUIRE(A(is...) == Approx(B(is...) + C(is...)));
        },
        2, 2);
  }
  SECTION(
      "A(i, k) = B(i, j) * C(j, k) + D(i, j) * E(j, k) - matrix-matrix "
      "addition") {
    A(i, j)      = B(i, j) * C(j, k) + B(i, j) * C(j, k) + B(i, j) * C(j, k);
    auto const F = B * C;

    for_loop(
        [&](auto const... is) { REQUIRE(A(is...) == Approx(3 * F(is...))); }, 2,
        2);
  }
  SECTION("A(i) = B(j, j, i)") {
    auto A = vec2{};
    auto B = tensor<real_type, 2, 2, 2>::randu();
    A(i)   = B(j, j, i);
    REQUIRE(A(0) == B(0, 0, 0) + B(1, 1, 0));
    REQUIRE(A(1) == B(0, 0, 1) + B(1, 1, 1));
  }

  SECTION("A(i, j) = b(i) * b(j) - outer product") {
    auto A = mat2{};
    auto b  = vec2::randu();
    A(i, j) = b(i) * b(j);
    REQUIRE(A(0, 0) == b(0) * b(0));
    REQUIRE(A(0, 1) == b(0) * b(1));
    REQUIRE(A(1, 0) == b(1) * b(0));
    REQUIRE(A(1, 1) == b(1) * b(1));
  }
  SECTION("s = b(i) * b(i) - inner product") {
    auto   b = vec2::randu();
    real_type s = b(i) * b(i);
    REQUIRE(s == b(0) * b(0) + b(1) * b(1));
  }
}
//==============================================================================
}  // namespace tatooine::einstein_notation::test
//==============================================================================
