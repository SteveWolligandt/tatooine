#if TATOOINE_YAVIN_AVAILABLE
//#include <tatooine/analytical/fields/numerical/autonomous_particles_test_field.h>
#include <tatooine/analytical/fields/numerical/sinuscosinus.h>
#include <tatooine/analytical/fields/numerical/center.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/gpu/lic.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_lic_doublegyre", "[gpu][dg][doublegyre][lic]") {
  analytical::fields::numerical::doublegyre v;
  gpu::lic(v, linspace{0.0, 2.0, 2000}, linspace{0.0, 1.0, 1000}, 1.0,
           vec<size_t, 2>{1000, 500}, 30, 0.001)
      .write_png("dg_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_saddle", "[gpu][saddle][lic]") {
  analytical::fields::numerical::saddle v;
  gpu::lic(v, linspace{-1.0, 1.0, 500}, linspace{-1.0, 1.0, 500}, 0.0,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("saddle_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_center", "[gpu][center][lic]") {
  analytical::fields::numerical::center v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("center_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_sinuscosinus", "[gpu][sinuscosinus][lic]") {
  analytical::fields::numerical::sinuscosinus v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("sinuscosinus_lic_0.png");
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, M_PI/2,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("sinuscosinus_lic_pi2.png");
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, M_PI/4,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("sinuscosinus_lic_pi4.png");
}
//==============================================================================
TEST_CASE("gpu_lic_cosinussinus", "[gpu][cosinussinus][lic]") {
  analytical::fields::numerical::cosinussinus v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("cosinussinus_lic_0.png");
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, M_PI/2,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("cosinussinus_lic_pi2.png");
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, M_PI/4,
           vec<size_t, 2>{1000, 1000}, 30, 0.001)
      .write_png("cosinussinus_lic_pi4.png");
}
//==============================================================================
//TEST_CASE("gpu_lic_autonomous_particles_test_field",
//          "[gpu][autonomous_particles_test_field][lic]") {
//  analytical::fields::numerical::autonomous_particles_test_field v;
//  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
//           vec<size_t, 2>{1000, 1000}, 30, 0.001)
//      .write_png("autonomous_particles_test_field.png");
//}
//==============================================================================
}  // namespace tatooine::gpu::test
//==============================================================================
#endif
