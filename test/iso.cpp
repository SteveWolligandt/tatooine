#include <tatooine/analytical/numerical/abcflow.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/isolines.h>
#include <tatooine/isosurface.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
TEST_CASE("isosurface_abcflow_length", "[iso][isosurface][numerical][abcflow]") {
  isosurface(length(analytical::numerical::abcflow{}),
                grid{linspace{-10.0, 10.0, 256},
                     linspace{-10.0, 10.0, 256},
                     linspace{-10.0, 10.0, 256}},
                1)
      .write_vtk("isosurface_abc.vtk");
}
//------------------------------------------------------------------------------
TEST_CASE("isolines_doublegyre_length", "[iso][isolines][numerical][doublegyre]") {
  auto const l = length(analytical::numerical::doublegyre{});
  SECTION("field") {
    write_vtk(
        isolines(l, grid{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100}},
                 0.1),
        "isolines_doublegyre.vtk");
  }
  SECTION("sampled") {
    grid g{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100}};
    auto& s = g.sample_to_vertex_property(
        [&](auto const& x) { return l(x, 0); }, "l");
    g.write_vtk("magnitude_doublegyre.vtk");
    write_vtk(isolines(s, 0.1), "isolines_doublegyre_sampled.vtk");
  }
}
//------------------------------------------------------------------------------
TEST_CASE("isosurface_random",
          "[iso][isosurface][static_multidim_arra][random]") {
  auto data = dynamic_multidim_array<double, x_fastest>::rand(
      random::uniform{-1.0, 1.0, std::mt19937{1234}}, 20, 20, 20);
  isosurface(
      data, axis_aligned_bounding_box{vec3{-10, -10, -10}, vec3{10, 10, 10}}, 0)
      .write_vtk("isosurface_random.vtk");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
