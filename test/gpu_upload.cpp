#include <tatooine/doublegyre.h>
#include <tatooine/multidim_array.h>
#include <tatooine/gpu/upload.h>

#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_upload_doublegyre", "[gpu][dg][doublegyre][upload]") {
  numerical::doublegyre<float> v;
  linspace<float>              xdomain{0.0, 2.0, 21};
  linspace<float>              ydomain{0.0, 1.0, 11};
  float                        t     = 0;
  auto                         v_gpu = gpu::upload(v, xdomain, ydomain, t);
  dynamic_multidim_array<vec<double, 2>> v_cpu(21, 11);
  v_gpu.download_data(reinterpret_cast<float*>(v_cpu.data_ptr()));

  for (size_t iy = 0; iy < ydomain.size(); ++iy) {
    for (size_t ix = 0; ix < xdomain.size(); ++ix) {
      CAPTURE(ix, iy, v_cpu(ix, iy), v({xdomain[ix], ydomain[iy]}, t));
      REQUIRE(approx_equal(v_cpu(ix, iy), v({xdomain[ix], ydomain[iy]}, t)));
    }
  }
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
