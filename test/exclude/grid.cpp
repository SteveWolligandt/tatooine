#include <tatooine/grid.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

//TEST_CASE("grid_mutate_seq_straight_prev_at",
//          "[grid][mutate][prev][straight]") {
//  const size_t new_prev_size           = 1;
//  const size_t elem_pos                = 2;
//  const size_t initial_sequence_length = 3;
//
//  grid            g{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
//  std::mt19937_64 rand{123};
//  std::mt19937    r2{std::random_device{}()};
//  auto            seq = g.random_vertex_sequence(initial_sequence_length, rand);
//  g.mutate_seq_straight_prev_at(seq, next(begin(seq), elem_pos), 130,
//                                new_prev_size, r2);
//  for (const auto& v : seq) { std::cerr << v.position() << '\n'; }
//  std::cerr << '\n';
//  REQUIRE(initial_sequence_length - elem_pos + new_prev_size == seq.size());
//  for (const auto& v : seq) { std::cerr << v.position() << '\n'; }
//}
//
////==============================================================================
//TEST_CASE("grid_mutate_seq_straight_next_at",
//          "[grid][mutate][next][straight]") {
//  const size_t new_next_size           = 2;
//  const size_t elem_pos                = 1;
//  const size_t initial_sequence_length = 3;
//
//  grid            g{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
//  std::mt19937_64 rand{123};
//  std::mt19937    r2{std::random_device{}()};
//  auto            seq = g.random_vertex_sequence(initial_sequence_length, rand);
//  for (const auto& v : seq) { std::cerr << v.position() << '\n'; }
//  std::cerr << '\n';
//  g.mutate_seq_straight_next_at(seq, next(begin(seq), elem_pos), 130,
//                                new_next_size, r2);
//  for (const auto& v : seq) { std::cerr << v.position() << '\n'; }
//}
//
////==============================================================================
//TEST_CASE("grid_mutate_seq_straight",
//          "[grid][mutate][straight]") {
//  grid            g{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
//  std::mt19937_64 rand{123};
//  std::mt19937    r2{std::random_device{}()};
//  auto            seq = g.random_vertex_sequence(5, rand);
//  for (const auto& v : seq) { std::cerr << v.position() << '\n'; }
//  std::cerr << '\n';
//  auto mseq = g.mutate_seq_straight(seq, next(begin(seq), 2), 130, 2, r2);
//  for (const auto& v : mseq) { std::cerr << v.position() << '\n'; }
//}

////==============================================================================
//TEST_CASE("grid_remove_dim",
//          "[grid][reduce]") {
//  grid g{linspace{0.0, 1.0, 3}, linspace{0.0, 2.0, 4}, linspace{0.0, 3.0, 5},
//         linspace{0.0, 4.0, 6}};
//  auto rg = g.remove_dimension(1, 2);
//  REQUIRE(rg.num_dimensions() == 2);
//  REQUIRE(rg.dimension(0).back() == 1.0);
//  REQUIRE(rg.dimension(1).back() == 4.0);
//}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
