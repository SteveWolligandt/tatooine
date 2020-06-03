#include <tatooine/doublegyre.h>
#include <tatooine/saddle.h>
#include <tatooine/gpu/lic.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_lic_doublegyre", "[gpu][dg][doublegyre][lic]") {
  numerical::doublegyre v;
  gpu::lic(v, linspace{0.0, 2.0, 2001}, linspace{0.0, 1.0, 1001}, 0.0,
           vec<size_t, 2>{1000, 500}, 10, 0.01)
      .write_png("dg_lic.png");
}
//==============================================================================
TEST_CASE("gpu_lic_saddle", "[gpu][dg][saddle][lic]") {
  numerical::saddle v;
  gpu::lic(v, linspace{-1.0, 1.0, 501}, linspace{-1.0, 1.0, 501}, 0.0,
           vec<size_t, 2>{500, 500}, 10, 0.01)
      .write_png("saddle_lic.png");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
