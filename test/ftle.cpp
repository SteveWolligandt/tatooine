#include <tatooine/analytical/fields/numerical/counterexample_sadlo.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/modified_doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/for_loop.h>
#include <tatooine/ftle_field.h>
#include <tatooine/grid.h>
#include <tatooine/ode/boost/rungekuttafehlberg78.h>
#include <tatooine/linspace.h>

#include <catch2/catch.hpp>
//==============================================================================
namespace tatooine::test {
//==============================================================================
using namespace tatooine::analytical::fields::numerical;
template <typename V, typename Grid, typename T0, typename Tau>
void ftle_test_vectorfield(V const&, Grid&, T0, Tau, std::string const&);
//------------------------------------------------------------------------------
template <typename FlowmapGradient, typename Grid, typename T0, typename Tau>
void ftle_test_flowmap_gradient(FlowmapGradient const& flowmap_gradient,
                                Grid& ftle_grid, T0 t0, Tau tau,
                                std::string const& path);
//==============================================================================
TEST_CASE("ftle_doublegyre", "[ftle][doublegyre][dg]") {
  doublegyre v;
  v.set_infinite_domain(true);
  grid   sample_grid{linspace{0.0, 2.0, 200}, linspace{0.0, 1.0, 100}};
  double t0 = 0, tau = -10;
  SECTION("direct") {
    ftle_test_vectorfield(
        v, sample_grid, t0, tau,
        "dg_ftle_" + std::to_string(t0) + "_" + std::to_string(tau));
  }
}
//==============================================================================
TEST_CASE("ftle_modified_doublegyre", "[ftle][modified_doublegyre][mdg]") {
  modified_doublegyre v;
  grid sample_grid{linspace{0.0, 2.0, 200}, linspace{-1.0, 1.0, 100}};
  for (auto t0 : linspace(1.0, 3.0, 3)) {
    for (auto tau : std::array{-20, 20}) {
      ftle_test_vectorfield(
          v, sample_grid, t0, tau,
          "ndg_ftle_" + std::to_string(tau) + "_" + std::to_string(t0));
    }
  }
}
//==============================================================================
TEST_CASE("ftle_counterexample_sadlo", "[ftle][counterexample_sadlo]") {
  counterexample_sadlo v;
  grid sample_grid{linspace{-3.0, 3.0, 200}, linspace{-3.0, 3.0, 200}};
  ftle_test_vectorfield(v, sample_grid, 0.0, 10.0,
                        "ftle_counterexample_sadlo_ftle_tau_" +
                            std::to_string(10.0) + "_t0_" +
                            std::to_string(0.0));
}
//==============================================================================
TEST_CASE("ftle_saddle", "[ftle][saddle]") {
  saddle                                   v;
  grid<linspace<double>, linspace<double>> ftle_grid{linspace{-1.0, 1.0, 200},
                                                     linspace{-1.0, 1.0, 200}};
  auto& ftle_prop = ftle_grid.add_vertex_property<double>("ftle");
  auto const t0   = 0;
  auto const tau  = 10;

  ftle_field f{v, tau};
  for_loop(
      [&](auto const... is) {
        ftle_prop(is...) = f(vec{ftle_grid.vertex_at(is...)}, t0);
      },
      200, 200);
  ftle_grid.write_vtk("ftle_saddle_tau" + std::to_string(tau) + "_t0_" +
                      std::to_string(t0) + ".vtk");
}
//==============================================================================
template <typename V, typename Grid, typename T0, typename Tau>
void ftle_test_vectorfield(V const& v, Grid& ftle_grid, T0 t0, Tau tau,
                           std::string const& path) {
  auto& ftle_prop =
      ftle_grid.template add_vertex_property<double>(
          "ftle");
  //ftle_field f{v, tau, ode::vclibs::rungekutta43<double, 2>{}};
  ftle_field f{v, tau, ode::boost::rungekuttafehlberg78<double, 2>{}};
  f.flowmap_gradient().set_epsilon(
      vec{(ftle_grid.template back<0>() - ftle_grid.template front<0>()) /
              static_cast<double>(ftle_grid.template size<0>()),
          (ftle_grid.template back<1>() - ftle_grid.template front<1>()) /
              static_cast<double>(ftle_grid.template size<1>())});
  for_loop(
      [&](auto const... is) {
        vec const x{ftle_grid.vertex_at(is...)};
        ftle_prop(is...) = f(x, t0);
      },
      ftle_grid.template size<0>(), ftle_grid.template size<1>());
  ftle_grid.write_vtk(path + ".vtk");
  ftle_prop.write_png(path + ".png");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
