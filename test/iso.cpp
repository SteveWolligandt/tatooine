#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/field_operations.h>
#include <tatooine/isolines.h>
#include <tatooine/isosurface.h>

#include <catch2/catch_test_macros.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("isosurface_abcflow_length", "[iso][isosurface][numerical][abcflow]") {
  isosurface(
      euclidean_length(analytical::numerical::abcflow{}),
      rectilinear_grid{linspace{-10.0, 10.0, 10}, linspace{-10.0, 10.0, 10},
                       linspace{-10.0, 10.0, 10}},
      1);
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
