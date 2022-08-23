#include <tatooine/doublegyre.h>
#include <tatooine/curvature_field.h>
#include <tatooine/grid_sampler.h>
#include <catch2/catch_test_macros.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("curvature_field_1", "[curvature_field][numerical][doublegyre]") {
  numerical::doublegyre dg;
  auto                  cv_dg = curvature(dg);
  resample<interpolation::linear, interpolation::linear>(
      cv_dg, grid{linspace{0.0, 2.0, 1000}, linspace{0.0, 1.0, 500}}, 0)
      .sampler().write_vtk("curvature_doublegyre.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
