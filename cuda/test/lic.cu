#include <tatooine/cuda/lic.cuh>
#include <tatooine/doublegyre.h>
#include <tatooine/modified_doublegyre.h>
#include <tatooine/write_png.h>

#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine {
namespace cuda {
namespace test {
//==============================================================================
TEST_CASE("lic_dg", "[lic][dg]") {
  const numerical::doublegyre<float> v;
  for (auto t : linspace<float>(0, 10, 11)) {
    auto lic_tex = call_lic_kernel(v, linspace<double>(0, 2, 2000),
                                   linspace<double>(0, 1, 1000), t, 100, 0.001,
                                   std::mt19937{1234});
    write_png<1>("lic_dg_" + std::to_string(t) + ".png", lic_tex.download(),
                 2000, 1000);
  }
}
TEST_CASE("lic_mdg", "[lic][mdg]") {
  const numerical::modified_doublegyre<float> v;
  for (auto t : linspace<float>(0, 10, 11)) {
    auto lic_tex = call_lic_kernel(v, linspace<double>(0, 2, 2000),
                                   linspace<double>(0, 1, 1000), t, 100, 0.001,
                                   std::mt19937{1234});
    write_png<1>("lic_mdg_" + std::to_string(t) + ".png", lic_tex.download(),
                 2000, 1000);
  }
}
//==============================================================================
}  // namespace test
}  // namespace cuda
}  // namespace tatooine
//==============================================================================
