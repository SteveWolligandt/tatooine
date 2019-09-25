#include <tatooine/grid.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("grid_mutate_seq_straight_prev_at", "[grid][mutate][straight]") {
  grid g{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
  std::mt19937_64 rand{123};
  auto seq = g.random_vertex_sequence(3, rand);

  g.mutate_seq_straight_prev_at(seq, next(begin(seq)), 130, 3, rand);
  for (const auto& v : seq) { std::cerr << v.position() << '\n'; }
}


//==============================================================================
}  // namespace tatooine::test
//==============================================================================
