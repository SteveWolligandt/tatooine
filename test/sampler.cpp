#include<tatooine/sampler.h>
#include<tatooine/grid.h>
#include<catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("sampler_contiguous_array", "[sampler][contiguous_array]") {
  grid g{linspace{0.0, 10.0, 11}, linspace{0.0, 10.0, 11}};
  using container_t = dynamic_multidim_array<double>;
  using grid_t = decltype(g);
  using interpolation_kernel = interpolation::linear<double>;
  using sampler_t =
      sampler<grid_t, container_t, interpolation_kernel, interpolation_kernel>;
  sampler_t s{g, 11, 11};
  s.container()(0, 0) = 1;
  s.container()(1, 0) = 3;
  s.container()(0, 1) = 2;
  s.container()(1, 1) = 4;

  REQUIRE(s.container().at(0, 0) == 1);
  REQUIRE(s.container().at(1, 0) == 3);
  REQUIRE(s.sample(0.1, 0) == Approx(0.9 * 1 + 0.1 * 3));
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
