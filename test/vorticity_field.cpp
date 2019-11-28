#include <tatooine/doublegyre.h>
#include <tatooine/vorticity_field.h>
#include <tatooine/spacetime_field.h>
#include <tatooine/grid_sampler.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("vorticity_field_1", "[curvature_field][numerical][doublegyre]") {
  numerical::doublegyre dg;
  auto                  vort_length_dg = length(vorticity(spacetime(dg)));
  resample<interpolation::linear, interpolation::linear, interpolation::linear>(
      spacetime(dg),
      grid{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100},
           linspace{0.0, 10.0, 60}},
      0)
      .sampler()
      .write_vtk("doublegyre.vtk");
  resample<interpolation::linear, interpolation::linear, interpolation::linear>(
      vort_length_dg,
      grid{linspace{0.0, 2.0, 100}, linspace{0.0, 1.0, 50},
           linspace{0.0, 10.0, 30}},
      0)
      .sampler()
      .write_vtk("vorticity_length_doublegyre.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
