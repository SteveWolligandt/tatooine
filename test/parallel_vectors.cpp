#include <catch2/catch.hpp>

#include <tatooine/analytical/fields/numerical/counterexample_sadlo.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/analytical/fields/numerical/modified_doublegyre.h>
#include <tatooine/parallel_vectors.h>
#include <tatooine/spacetime_vectorfield.h>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("parallel_vectors_pv_on_tri", "[parallel_vectors][pv][pv_on_tri]") {
  auto x = detail::pv_on_tri(
      vec{0.0, 0.0, 0.0}, vec{-0.5, -1.0, -0.5}, vec{-0.5, 1.0, 0.5},
      vec{1.0, 0.0, 0.0}, vec{ 0.5, -1.0, -0.5}, vec{ 0.5, 1.0, 0.5},
      vec{1.0, 1.0, 1.0}, vec{ 0.8,  1.0, -0.7}, vec{ 1.5, 3.0, 0.5});
  CAPTURE(x);
  REQUIRE(x);
  REQUIRE(approx_equal(*x, vec{0.5, 0.0, 0.0}));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
