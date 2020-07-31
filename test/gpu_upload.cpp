#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/gpu/upload.h>
#include <tatooine/multidim_array.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_upload_doublegyre", "[gpu][dg][doublegyre][upload]") {
  analytical::fields::numerical::doublegyre<float> v;
  linspace<float>              xdomain{0.0, 2.0, 21};
  linspace<float>              ydomain{0.0, 1.0, 11};
  float                        t     = 0;
  dynamic_multidim_array<vec<float, 2>> v_sampled(xdomain.size(), ydomain.size());
  size_t                                yi = 0;
  for (auto y : ydomain) {
    size_t xi = 0;
    for (auto x : xdomain) {
      v_sampled(xi, yi) = v(vec{x, y}, t);
      ++xi;
    }
    ++yi;
  }
  auto                                   v_gpu = gpu::upload(v_sampled);
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
