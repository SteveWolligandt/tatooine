#if TATOOINE_YAVIN_AVAILABLE
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/gpu/upload.h>
#include <tatooine/multidim_array.h>
#include <tatooine/demangling.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::gpu::test {
//==============================================================================
TEST_CASE("gpu_upload_doublegyre", "[gpu][dg][doublegyre][upload]") {
  analytical::fields::numerical::doublegyre<float> v;
  linspace<float>              xdomain{0.0, 2.0, 21};
  linspace<float>              ydomain{0.0, 1.0, 11};
  float                        t     = 0;
  
  // upload
  auto v_gpu = gpu::upload_tex(sample_to_vector(v, grid{xdomain, ydomain}, t), 21, 11);
  REQUIRE(v_gpu.width() == xdomain.size());
  REQUIRE(v_gpu.height() == ydomain.size());
  CAPTURE(type_name(v_gpu));
  REQUIRE(std::is_same_v<decltype(v_gpu), yavin::tex2rg<float>>);

  // download
  auto v_cpu = download(v_gpu);
  REQUIRE(v_cpu.size(0) == xdomain.size());
  REQUIRE(v_cpu.size(1) == ydomain.size());

  size_t i = 0;
  for (size_t iy = 0; iy < ydomain.size(); ++iy) {
    for (size_t ix = 0; ix < xdomain.size(); ++ix) {
      CAPTURE(ix, iy, v({xdomain[ix], ydomain[iy]}, t), v_cpu(ix, iy));
      REQUIRE(approx_equal(v_cpu(ix, iy), v({xdomain[ix], ydomain[iy]}, t)));
      ++i;
    }
  }
}
////==============================================================================
//TEST_CASE("gpu_upload_doublegyre_resample", "[gpu][dg][doublegyre][upload]") {
//  analytical::fields::numerical::doublegyre<float> v;
//  linspace<float>              xdomain{0.0, 2.0, 21};
//  linspace<float>              ydomain{0.0, 1.0, 11};
//  float                        t     = 0;
//  auto [sample_grid, name] = resample(v, grid{xdomain, ydomain}, t);
//  auto& v_sampled = sample_grid.vertex_property<decltype(v)::tensor_t>(name);
//  REQUIRE(sample_grid.size<0>() == xdomain.size());
//  REQUIRE(sample_grid.size<1>() == ydomain.size());
//  REQUIRE(&sample_grid == &v_sampled.grid());
//  REQUIRE(v_sampled.size<0>() == xdomain.size());
//  REQUIRE(v_sampled.size<1>() == ydomain.size());
//  auto  v_gpu     = gpu::upload(v_sampled);
//  //CAPTURE(type_name(v_gpu));
//  //REQUIRE(std::is_same_v<decltype(v_gpu), yavin::tex2rg<float>>);
//  //auto v_cpu = download(v_gpu);
//  //
//  //for (size_t iy = 0; iy < ydomain.size(); ++iy) {
//  //  for (size_t ix = 0; ix < xdomain.size(); ++ix) {
//  //    CAPTURE(ix, iy, v_sampled.data_at(ix, iy), v_cpu(ix, iy));
//  //    REQUIRE(approx_equal(v_cpu(ix, iy), v_sampled.data_at(ix, iy)));
//  //  }
//  //}
//}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
#endif
