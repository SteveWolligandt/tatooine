#include <catch2/catch_test_macros.hpp>
#include <tatooine/vtk_legacy.h>

//==============================================================================
namespace tatooine::vtk::test {
//==============================================================================

TEST_CASE("vtk_legacy1", "[vtk_legacy]"){
  vtk::legacy_file_writer f{"test.vtk", POLYDATA};
}

//==============================================================================
}  // namespace tatooine::vtk::test
//==============================================================================
