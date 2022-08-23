#include <tatooine/doublegyre.h>
#include <tatooine/helicity_field.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/grid_sampler.h>
#include <catch2/catch_test_macros.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("helicity_field_1", "[helicity_field][numerical][doublegyre]") {
  numerical::doublegyre vf;
  auto                  stvf   = spacetime(vf);
  auto                  lvstvf = length(helicity(stvf));

  grid sample_grid{linspace{0.0,  2.0, 200},
                   linspace{0.0,  1.0, 100},
                   linspace{0.0, 10.0, 60}};

  resample<interpolation::linear, interpolation::linear, interpolation::linear>(
      stvf, sample_grid, 0)
      .sampler() 
      .write_vtk("doublegyre.vtk");

  resample<interpolation::linear, interpolation::linear, interpolation::linear>(
      lvstvf, sample_grid, 0)
      .sampler()
      .write_vtk("helicity_length_doublegyre.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
