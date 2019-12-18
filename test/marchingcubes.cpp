#include <tatooine/marchingcubes.h>
#include <tatooine/abcflow.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("marchingcubes_1", "[marchingcubes][numerical][abcflow]") {
  marchingcubes(length(numerical::abcflow{}),
                grid{linspace{-10.0, 10.0, 256}, linspace{-10.0, 10.0, 256},
                     linspace{-10.0, 10.0, 256}},
                1)
      .write_vtk("mc_abc.vtk");
}
//------------------------------------------------------------------------------
TEST_CASE("marchingcubes_2", "[marchingcubes][static_multidim_arra][random]") {
  auto data = dynamic_multidim_array<double, x_fastest>::rand(
      random_uniform{0.0, 1.0, std::mt19937{1234}}, 20, 20, 20);
  marchingcubes(
      data, boundingbox{vec{-10.0, -10.0, -10.0}, vec{10.0, 10.0, 10.0}},
      0.1)
      .write_vtk("mc_random.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
