#include <tatooine/einstein_notation.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::einstein_notation::test {
//==============================================================================
TEST_CASE("einstein_notation_free_indices",
          "[einstein_notation][free_indices]") {
  using indices = free_indices<i_t, i_t, j_t, k_t>;
  REQUIRE(static_set_size<indices> == 2);
  REQUIRE(static_set_includes<indices, j_t>);
  REQUIRE(static_set_includes<indices, k_t>);
}
//==============================================================================
TEST_CASE("einstein_notation_contracted_indices",
          "[einstein_notation][contracted_indices]") {
  using indices = contracted_indices<i_t, i_t, j_t, k_t>;
  REQUIRE(static_set_size<indices> == 1);
  REQUIRE(static_set_includes<indices, i_t>);
}
//==============================================================================
}  // namespace tatooine::einstein_notation::test
//==============================================================================
