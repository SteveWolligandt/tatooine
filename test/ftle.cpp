#include <tatooine/counterexample_sadlo.h>
#include <tatooine/doublegyre.h>
#include <tatooine/newdoublegyre.h>
#include <tatooine/ftle.h>
#include <tatooine/grid_sampler.h>
#include <tatooine/integration/boost/rungekutta4.h>
#include <tatooine/integration/boost/rungekuttacashkarp54.h>
#include <tatooine/integration/boost/rungekuttadopri5.h>
#include <tatooine/integration/boost/rungekuttafehlberg78.h>
#include <tatooine/integration/vclibs/rungekutta43.h>
#include <tatooine/linspace.h>
#include <catch2/catch.hpp>

//==============================================================================
namespace tatooine::test {
//==============================================================================

template <typename V, typename Grid, typename T0, typename Tau, typename Integrator>
void ftle_test(const V&, const Grid&, T0, Tau, const Integrator&,
               const std::string&);
//using Integrator = integration::vclibs::rungekutta43<double, 2>;
//using Integrator = integration::boost::rungekuttafehlberg78<double, 2>;
using Integrator = integration::boost::rungekutta4<double, 2>;
//using Integrator = integration::boost::rungekuttadopri5<double, 2>;
//using Integrator = integration::boost::rungekuttacashkarp54<double, 2>;

//==============================================================================
TEST_CASE("ftle_doublegyre", "[ftle][doublegyre][dg]") {
  numerical::doublegyre v;
  grid sample_grid{linspace{0.0, 2.0, 1000}, linspace{0.0, 1.0, 500}};
  double t0 = 0, tau = 10;
  ftle_test(v, sample_grid, t0, tau, Integrator{},
            "dg_ftle_" + std::to_string(t0) + "_" + std::to_string(tau));
}
//==============================================================================
TEST_CASE("ftle_newdoublegyre", "[ftle][newdoublegyre][ndg]") {
  numerical::newdoublegyre v;
  grid sample_grid{linspace{0.0, 2.0, 1000}, linspace{-1.0, 1.0, 1000}};
  for (auto tau : std::array{-20, 20}) {
    for (auto t0 : linspace(-10.0, 10.0, 21)) {
      ftle_test(v, sample_grid, t0, tau, Integrator{},
                "ndg_ftle_" + std::to_string(tau) + "_" + std::to_string(t0));
    }
  }
}

//==============================================================================
TEST_CASE("ftle_counterexample_sadlo", "[ftle][counterexample_sadlo]") {
  numerical::counterexample_sadlo v;
  grid sample_grid{linspace{-3.0, 3.0, 2000}, linspace{-3.0, 3.0, 2000}};
  for (auto tau : std::array{-20, 20}) {
    for (auto t0 : linspace(-10.0, 10.0, 21)) {
      ftle_test(v, sample_grid, t0, tau, Integrator{},
                "counterexample_sadlo_ftle_" + std::to_string(tau) + "_" +
                    std::to_string(t0));
    }
  }
}

//==============================================================================
template <typename V, typename Grid, typename T0, typename Tau,
          typename Integrator>
void ftle_test(const V& v, const Grid& sample_grid, T0 t0, Tau tau,
               const Integrator& integrator, const std::string& path) {
  ftle v_ftle{v, integrator, tau};
  v_ftle.set_eps(
      vec{(sample_grid.dimension(0).back() - sample_grid.dimension(0).front()) /
              static_cast<double>(sample_grid.dimension(0).size()),
          (sample_grid.dimension(1).back() - sample_grid.dimension(1).front()) /
              static_cast<double>(sample_grid.dimension(1).size())});
  auto ftle_grid = resample<interpolation::linear, interpolation::linear>(
      v_ftle, sample_grid, t0);
  ftle_grid.sampler().write_vtk(path + ".vtk");
}

//==============================================================================
}  // namespace tatooine::test
//==============================================================================
