#include <tatooine/marchingcubes.h>
#include <tatooine/abcflow.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

TEST_CASE("marchingcubes_1",
          "[marchingcubes][numerical][abcflow]") {
  marchingcubes(length(numerical::abcflow{}),
                grid{linspace{-10.0, 10.0, 256}, linspace{-10.0, 10.0, 256},
                     linspace{-10.0, 10.0, 256}},
                1)
      .write_vtk("mc_abc.vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
