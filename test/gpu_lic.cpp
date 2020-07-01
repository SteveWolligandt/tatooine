#include <tatooine/autonomous_particles_test_field.h>
#include <tatooine/center_field.h>
#include <tatooine/doublegyre.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/saddle_field.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_lic_doublegyre", "[gpu][dg][doublegyre][lic]") {
  numerical::doublegyre v;
  gpu::lic(v, linspace{0.0, 2.0, 2001}, linspace{0.0, 1.0, 1001}, 0.0,
           vec<size_t, 2>{1000, 500}, 30, 0.001)
      .write_png("dg_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_saddle", "[gpu][saddle][lic]") {
  numerical::saddle_field v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("saddle_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_center", "[gpu][center][lic]") {
  numerical::center_field v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("center_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_autonomous_particles_test_field",
          "[gpu][autonomous_particles_test_field][lic]") {
  numerical::autonomous_particles_test_field v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("autonomous_particles_test_field.png");
}
//==============================================================================
}  // namespace tatooine::gpu::test
//==============================================================================
