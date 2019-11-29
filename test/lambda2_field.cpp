#include <tatooine/doublegyre.h>
#include <tatooine/lambda2_field.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/grid_sampler.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("lambda2_field_1", "[lambda2_field][numerical][doublegyre]") {
  numerical::doublegyre vf;
  auto                  stvf   = spacetime(vf);
  auto                  l2stvf = lambda2(stvf);

  grid sample_grid{linspace{0.0,  2.0, 200},
                   linspace{0.0,  1.0, 100},
                   linspace{0.0, 10.0, 60}};

  resample<interpolation::linear, interpolation::linear, interpolation::linear>(
      stvf, sample_grid, 0)
      .sampler() 
      .write_vtk("doublegyre.vtk");

  resample<interpolation::linear, interpolation::linear, interpolation::linear>(
      l2stvf, sample_grid, 0)
      .sampler()
      .write_vtk("lambda2_doublegyre.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
