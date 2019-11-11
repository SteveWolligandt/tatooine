#include <tatooine/abcflow.h>
#include <tatooine/doublegyre.h>

#include <catch2/catch.hpp>
#include <tatooine/cuda/buffer.cuh>
#include <tatooine/cuda/coordinate_conversion.cuh>
#include <tatooine/cuda/field.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================
__global__ void kernel2_steady(steady_vectorfield<float, 2, 2> v,
                               buffer<float> vf_out, buffer<float> pos_out,
                               buffer<unsigned int> idx_out) {
  const auto res       = v.resolution();
  const auto globalIdx = make_uint2(blockIdx.x * blockDim.x + threadIdx.x,
                                    blockIdx.y * blockDim.y + threadIdx.y);
  if (globalIdx.x >= res.x || globalIdx.y >= res.y) { return; }

  // sample vectorfield
  const auto x = global_idx_to_domain_pos(globalIdx, v.min(), v.max(), res);
  const auto sample = v(x);

  // sample texture and assign to output array
  const size_t plainIdx     = globalIdx.x + globalIdx.y * res.x;
  pos_out[plainIdx * 2]     = x.x;
  pos_out[plainIdx * 2 + 1] = x.y;
  vf_out[plainIdx * 2]      = sample.x;
  vf_out[plainIdx * 2 + 1]  = sample.y;
  idx_out[plainIdx * 2]     = globalIdx.x;
  idx_out[plainIdx * 2 + 1] = globalIdx.y;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TEST_CASE("field_to_tex2_steady", "[cuda][field][dg][steady]") {
  // create vector field
  const numerical::doublegyre<float> v;

  // sampled vector field and upload to gpu
  const double           t = 0;
  const linspace<double> x{0, 2, 21};
  const linspace<double> y{0, 1, 11};
  auto                   d_v = upload(v, grid<double, 2>{x, y}, t);
  buffer<float>          d_vf_out(2 * x.size() * y.size());
  buffer<float>          d_pos_out(2 * x.size() * y.size());
  buffer<unsigned int>   d_idx_out(2 * x.size() * y.size());
  // call kernel
  const dim3 num_grids{32, 32};
  const dim3 num_threads(x.size() / num_grids.x + 1,
                         y.size() / num_grids.y + 1);
  kernel2_steady<<<num_threads, num_grids>>>(d_v, d_vf_out, d_pos_out,
                                             d_idx_out);

  // download data from gpu
  const auto h_vf_out  = d_vf_out.download();
  const auto h_pos_out = d_pos_out.download();
  const auto h_idx_out = d_idx_out.download();
  for (size_t i = 0; i < h_pos_out.size(); i += 2) {
    tatooine::vec<float, 2>        v_gpu{h_vf_out[i], h_vf_out[i + 1]};
    tatooine::vec<float, 2>        x_gpu{h_pos_out[i], h_pos_out[i + 1]};
    tatooine::vec<unsigned int, 2> idx{h_idx_out[i], h_idx_out[i + 1]};
    tatooine::vec<float, 2>        x_cpu{x[idx(0)], y[idx(1)]};
    auto                           v_cpu = v(x_gpu, t);
    INFO("expected x_gpu: " << x_cpu);
    INFO("x_gpu: " << x_gpu);
    INFO("idx: " << idx);
    INFO("cpu: " << v_cpu);
    INFO("gpu: " << v_gpu);
    REQUIRE(((x_cpu(0) == x_gpu(0)) && (x_cpu(1) == x_gpu(1))));
    REQUIRE(((v_gpu(0) == Approx(v_cpu(0)).margin(1e-6)) &&
             (v_gpu(1) == Approx(v_cpu(1)).margin(1e-6))));
  }
  free(d_vf_out, d_pos_out, d_idx_out, d_v);
}
//==============================================================================
__global__ void kernel3_steady(steady_vectorfield<float, 3, 3> v,
                               buffer<float> vf_out, buffer<float> pos_out,
                               buffer<unsigned int> idx_out) {
  const auto res = v.resolution();
  const auto globalIdx =
      make_vec<unsigned int>(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y,
                             blockIdx.z * blockDim.z + threadIdx.z);
  if (globalIdx.x >= res.x || globalIdx.y >= res.y || globalIdx.z >= res.z) {
    return;
  }

  // sample vectorfield
  const auto uvw = global_idx_to_uvw(globalIdx, res);
  const auto x   = global_idx_to_domain_pos(globalIdx, v.min(), v.max(), res);
  const auto sample = v.evaluate_uv(uvw);

  // sample texture and assign to output array
  const size_t plainIdx =
      globalIdx.x + globalIdx.y * res.x + globalIdx.z * res.x * res.y;
  pos_out[plainIdx * 3]     = x.x;
  pos_out[plainIdx * 3 + 1] = x.y;
  pos_out[plainIdx * 3 + 2] = x.z;
  vf_out[plainIdx * 3]      = sample.x;
  vf_out[plainIdx * 3 + 1]  = sample.y;
  vf_out[plainIdx * 3 + 2]  = sample.z;
  idx_out[plainIdx * 3]     = globalIdx.x;
  idx_out[plainIdx * 3 + 1] = globalIdx.y;
  idx_out[plainIdx * 3 + 2] = globalIdx.z;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TEST_CASE("field_to_tex3_steady", "[cuda][field][abc][steady]") {
  using host_real   = double;
  using device_real = float;
  using lin         = linspace<host_real>;
  using gr          = grid<host_real, 3>;
  // create vector field
  const numerical::abcflow<host_real> v;

  // sampled vector field and upload to gpu
  const host_real      t = 0;
  const lin            x{-1, 1, 11};
  const lin            y{-1, 1, 11};
  const lin            z{-1, 1, 11};
  auto                 d_v = upload(v, gr{x, y, z}, t);
  buffer<device_real>  d_vf_out(3 * x.size() * y.size() * z.size());
  buffer<device_real>  d_pos_out(3 * x.size() * y.size() * z.size());
  buffer<unsigned int> d_idx_out(3 * x.size() * y.size() * z.size());
  // call kernel
  const dim3 num_grids{8, 8, 8};
  const dim3 num_threads(x.size() / num_grids.x + 1, y.size() / num_grids.y + 1,
                         z.size() / num_grids.z + 1);
  kernel3_steady<<<num_threads, num_grids>>>(d_v, d_vf_out, d_pos_out,
                                             d_idx_out);

  // download data from gpu
  const auto h_vf_out  = d_vf_out.download();
  const auto h_pos_out = d_pos_out.download();
  const auto h_idx_out = d_idx_out.download();
  for (size_t i = 0; i < h_vf_out.size(); i += 3) {
    tatooine::vec<host_real, 3>    v_gpu{h_vf_out[i], h_vf_out[i + 1],
                                      h_vf_out[i + 2]};
    tatooine::vec<host_real, 3>    x_gpu{h_pos_out[i], h_pos_out[i + 1],
                                    h_pos_out[i + 2]};
    tatooine::vec<unsigned int, 3> idx{h_idx_out[i], h_idx_out[i + 1],
                                       h_idx_out[i + 2]};
    tatooine::vec<host_real, 3>    x_cpu{x[idx(0)], y[idx(1)], y[idx(2)]};
    auto                           v_cpu = v(x_gpu, t);
    INFO("expected Pos: " << x_cpu);
    INFO("Pos: " << x_gpu);
    INFO("idx: " << idx);
    INFO("CPU: " << v_cpu);
    INFO("GPU: " << v_gpu);
    REQUIRE(((x_cpu(0) == Approx(x_gpu(0)).margin(1e-6)) &&
             (x_cpu(1) == Approx(x_gpu(1)).margin(1e-6)) &&
             (x_cpu(2) == Approx(x_gpu(2)).margin(1e-6))));
    REQUIRE(((v_gpu(0) == Approx(v_cpu(0)).margin(1e-6)) &&
             (v_gpu(1) == Approx(v_cpu(1)).margin(1e-6)) &&
             (v_gpu(2) == Approx(v_cpu(2)).margin(1e-6))));
  }
  free(d_vf_out, d_pos_out, d_idx_out, d_v);
}
//==============================================================================
__global__ void kernel2_unsteady(unsteady_vectorfield<float, 2, 2> v,
                                 buffer<float> vf_out, buffer<float> pos_out,
                                 buffer<float>        t_out,
                                 buffer<unsigned int> idx_out) {
  const auto res       = v.resolution();
  const auto globalIdx =
      make_vec<unsigned int>(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y,
                             blockIdx.z * blockDim.z + threadIdx.z);
  if (globalIdx.x >= res.x || globalIdx.y >= res.y || globalIdx.z >= res.z) {
    return;
  }

  // sample vectorfield
  const auto uvw    = global_idx_to_uvw(globalIdx, res);
  const auto xt     = global_idx_to_domain_pos(globalIdx, v.min(), v.max(),
                                           v.tmin(), v.tmax(), res);
  const auto sample = v.evaluate_uv(uvw);

  // sample texture and assign to output array
  const size_t plainIdx =
      globalIdx.x + globalIdx.y * res.x + globalIdx.z * res.x * res.y;
  pos_out[plainIdx * 2]     = xt.x;
  pos_out[plainIdx * 2 + 1] = xt.y;
  t_out[plainIdx]           = xt.z;
  vf_out[plainIdx * 2]      = sample.x;
  vf_out[plainIdx * 2 + 1]  = sample.y;
  idx_out[plainIdx * 3]     = globalIdx.x;
  idx_out[plainIdx * 3 + 1] = globalIdx.y;
  idx_out[plainIdx * 3 + 2] = globalIdx.z;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TEST_CASE("field_to_tex2_unsteady", "[cuda][field][dg][unsteady]") {
  // create vector field
  const numerical::doublegyre<float> v;

  // sampled vector field and upload to gpu
  const linspace<double> x{0, 2, 21};
  const linspace<double> y{0, 1, 11};
  const linspace<double> t{0, 10, 11};
  auto                   d_v = upload(v, grid<double, 2>{x, y}, t);
  buffer<float>          d_vf_out(2 * x.size() * y.size() * t.size());
  buffer<float>          d_pos_out(2 * x.size() * y.size() * t.size());
  buffer<float>          d_t_out(x.size() * y.size() * t.size());
  buffer<unsigned int>   d_idx_out(3 * x.size() * y.size() * t.size());
  // call kernel
  const dim3 num_grids{8, 8, 8};
  const dim3 num_threads(x.size() / num_grids.x + 1, y.size() / num_grids.y + 1,
                         t.size() / num_grids.z + 1);
  kernel2_unsteady<<<num_threads, num_grids>>>(d_v, d_vf_out, d_pos_out, d_t_out,
                                             d_idx_out);
  // download data from gpu
  const auto h_vf_out  = d_vf_out.download();
  const auto h_pos_out = d_pos_out.download();
  const auto h_t_out   = d_t_out.download();
  const auto h_idx_out = d_idx_out.download();
  for (size_t i = 0; i < x.size() * y.size() * t.size(); ++i) {
    tatooine::vec<unsigned int, 3> idx{h_idx_out[i * 3], h_idx_out[i * 3 + 1],
                                       h_idx_out[i * 3 + 2]};

    tatooine::vec<float, 2> x_gpu{h_pos_out[i * 2], h_pos_out[i * 2 + 1]};
    tatooine::vec<double, 2> x_cpu{x[idx(0)], y[idx(1)]};

    auto t_gpu = h_t_out[i];
    auto t_cpu = t[idx(2)];

    tatooine::vec<float, 2> v_gpu{h_vf_out[i * 2], h_vf_out[i * 2 + 1]};
    auto                    v_cpu = v(x_cpu, t_cpu);

    INFO("idx: " << idx);
    INFO("x_cpu: " << x_cpu);
    INFO("x_gpu: " << x_gpu);
    INFO("t_cpu: " << t_cpu);
    INFO("t_gpu: " << t_gpu);
    INFO("v_cpu: " << v_cpu);
    INFO("v_gpu: " << v_gpu);
    REQUIRE(((x_gpu(0) == Approx(x_cpu(0)).margin(1e-6)) &&
             (x_gpu(1) == Approx(x_cpu(1)).margin(1e-6))));
    REQUIRE(t_gpu == Approx(t_cpu).margin(1e-6));
    REQUIRE(((v_gpu(0) == Approx(v_cpu(0)).margin(1e-6)) &&
             (v_gpu(1) == Approx(v_cpu(1)).margin(1e-6))));
  }
  free(d_vf_out, d_pos_out, d_t_out, d_idx_out, d_v);
}

//==============================================================================
}  // namespace test
}  // namespace cuda
}  // namespace tatooine
//==============================================================================
