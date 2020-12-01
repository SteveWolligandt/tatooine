#include<tatooine/sampler.h>
#include<tatooine/grid.h>
#include<catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("sampler_contiguous_array", "[sampler][contiguous_array]") {
  grid<linspace<double>, linspace<double>> g{linspace{0.0, 10.0, 11},
                                             linspace{0.0, 10.0, 11}};
  auto& prop = g.add_vertex_property<double>("prop");
  auto  s    = prop.sampler<interpolation::linear>();
  prop(0, 0) = 1;
  prop(1, 0) = 3;
  prop(0, 1) = 2;
  prop(1, 1) = 4;

  REQUIRE(s.sample(0.1, 0) == Approx(0.9 * 1 + 0.1 * 3));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
