#include <tatooine/doublegyre.h>
#include <tatooine/gpu/lic.h>

#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_lic_doublegyre", "[gpu][dg][doublegyre][lic]") {
  numerical::doublegyre v;
  write_png(gpu::lic(v, linspace{0.0, 2.0, 2001}, linspace{0.0, 1.0, 1001}, 0.0,
                     vec<size_t, 2>{2000, 1000}),
            "dg_lic.png");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
