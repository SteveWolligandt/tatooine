#include <tatooine/analytical/fields/numerical/counterexample_sadlo.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/modified_doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/ftle_field.h>
#include <tatooine/grid_sampler.h>
#include <tatooine/linspace.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace tatooine::analytical::fields::numerical;
template <typename V, typename Grid, typename T0, typename Tau>
void ftle_test(V const&, Grid const&, T0, Tau, std::string const&);
//==============================================================================
TEST_CASE("ftle_doublegyre", "[ftle][doublegyre][dg]") {
  doublegyre v;
  grid   sample_grid{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100}};
  double t0 = 0, tau = 10;
  ftle_test(v, sample_grid, t0, tau,
            "dg_ftle_" + std::to_string(t0) + "_" + std::to_string(tau));
}
//==============================================================================
TEST_CASE("ftle_modified_doublegyre", "[ftle][modified_doublegyre][mdg]") {
  modified_doublegyre v;
  grid sample_grid{linspace{0.0, 2.0, 200}, linspace{-1.0, 1.0, 100}};
  for (auto t0 : linspace(1.0, 3.0, 3)) {
    for (auto tau : std::array{-20, 20}) {
      ftle_test(v, sample_grid, t0, tau,
                "ndg_ftle_" + std::to_string(tau) + "_" + std::to_string(t0));
    }
  }
}
//==============================================================================
TEST_CASE("ftle_counterexample_sadlo", "[ftle][counterexample_sadlo]") {
  counterexample_sadlo v;
  grid sample_grid{linspace{-3.0, 3.0, 200}, linspace{-3.0, 3.0, 200}};
  ftle_test(v, sample_grid, 0.0, 10.0,
            "ftle_counterexample_sadlo_ftle_tau_" + std::to_string(10.0) + "_t0_" +
                std::to_string(0.0));
}
//==============================================================================
TEST_CASE("ftle_saddle", "[ftle][saddle]") {
  saddle v;
  grid sample_grid{linspace{-1.0, 1.0, 200}, linspace{-1.0, 1.0, 200}};
  auto const t0 = 0;
  auto const tau = 10;

  ftle_field f{v, tau};
  auto ftle_grid = resample<interpolation::linear, interpolation::linear>(
      f, sample_grid, t0);
  ftle_grid.sampler().write_vtk("ftle_saddle_tau" + std::to_string(tau) +
                                "_t0_" + std::to_string(t0) + ".vtk");
}
//==============================================================================
template <typename V, typename Grid, typename T0, typename Tau>
void ftle_test(V const& v, Grid const& sample_grid, T0 t0, Tau tau,
               std::string const& path) {
  ftle_field f{v, tau};
  f.flowmap_gradient().set_epsilon(
      vec{(sample_grid.dimension(0).back() - sample_grid.dimension(0).front()) /
              static_cast<double>(sample_grid.dimension(0).size()),
          (sample_grid.dimension(1).back() - sample_grid.dimension(1).front()) /
              static_cast<double>(sample_grid.dimension(1).size())});
  auto ftle_grid = resample<interpolation::linear, interpolation::linear>(
      f, sample_grid, t0);
  ftle_grid.sampler().write_vtk(path + ".vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
