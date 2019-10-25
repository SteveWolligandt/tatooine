#include <tatooine/doublegyre.h>
#include <tatooine/duffing_oscillator.h>
#include <tatooine/modified_doublegyre.h>
#include <tatooine/write_png.h>

#include <catch2/catch.hpp>
#include <tatooine/cuda/lic.cuh>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================
TEST_CASE("lic_doublegyre", "[lic][dg][doublegyre]") {
  const numerical::doublegyre<double> v;
  const size_t                        sample_res_x = 200, sample_res_y = 100;
  const size_t                        lic_res_x = 1000, lic_res_y = 500;
  for (auto t : linspace<float>(0, 10, 11)) {
    auto lic_tex =
        call_lic_kernel(v,
                        grid<double, 2>{linspace<double>(0, 2, sample_res_x),
                                        linspace<double>(0, 1, sample_res_y)},
                        grid<double, 2>{linspace<double>(0, 2, lic_res_x),
                                        linspace<double>(0, 1, lic_res_y)},
                        t, 100, 0.001, std::mt19937{1234});
    write_png<1>("lic_doublegyre_" + std::to_string(t) + ".png",
                 lic_tex.download(), lic_res_x, lic_res_y);
  }
}
//==============================================================================
TEST_CASE("lic_modified_doublegyre", "[lic][mdg][modified_doublegyre]") {
  const numerical::modified_doublegyre<double> v;
  for (auto t : linspace<float>(0, 10, 11)) {
    auto lic_tex =
        call_lic_kernel(v,
                        grid<double, 2>{linspace<double>(0, 2, 100),
                                        linspace<double>(0, 1, 50)},
                        grid<double, 2>{linspace<double>(0, 2, 1000),
                                        linspace<double>(0, 1, 500)},
                        t, 100, 0.001, std::mt19937{1234});
    write_png<1>("lic_modified_doublegyre_" + std::to_string(t) + ".png",
                 lic_tex.download(), 1000, 500);
  }
}
//==============================================================================
TEST_CASE("lic_duffing_oscillator", "[lic][fdo][forced_duffing_oscillator]") {
  const numerical::forced_duffing_oscillator<double> v{};
  for (auto t : linspace<float>(0, 2 * M_PI, 62)) {
    auto lic_tex =
        call_lic_kernel(v,
                        grid<double, 2>{linspace<double>(-2, 2, 200),
                                        linspace<double>(-1, 1, 100)},
                        grid<double, 2>{linspace<double>(-2, 2, 2000),
                                        linspace<double>(-1, 1, 1000)},
                        t, 100, 0.001, std::mt19937{1234});
    write_png<1>("lic_forced_duffing_oscillator_" + std::to_string(t) + ".png",
                 lic_tex.download(), 2000, 1000);
  }
}
//==============================================================================
}  // namespace test
}  // namespace cuda
}  // namespace tatooine
//==============================================================================
