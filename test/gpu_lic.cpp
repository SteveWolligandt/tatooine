#include <tatooine/doublegyre.h>
#include <tatooine/saddle_field.h>
#include <tatooine/circle_field.h>
#include <tatooine/gpu/lic.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_lic_doublegyre", "[gpu][dg][doublegyre][lic]") {
  numerical::doublegyre v;
  gpu::lic(v, linspace{0.0, 2.0, 2001}, linspace{0.0, 1.0, 1001}, 0.0,
           vec<size_t, 2>{1000, 500}, 20, 0.005)
      .write_png("dg_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_saddle", "[gpu][dg][saddle][lic]") {
  numerical::saddle_field v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{500, 500}, 20, 0.005)
      .write_png("saddle_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_circle", "[gpu][dg][circle][lic]") {
  numerical::circle_field v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{500, 500}, 20, 0.005)
      .write_png("circle_lic.png");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
