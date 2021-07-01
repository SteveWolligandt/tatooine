#include <tatooine/analytical/fields/numerical/counterexample_sadlo.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/analytical/fields/numerical/modified_doublegyre.h>
#include <tatooine/analytical/fields/numerical/saddle.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/for_loop.h>
#include <tatooine/ftle_field.h>
#include <tatooine/grid.h>
#include <tatooine/gpu/lic.h>
//#include <tatooine/ode/boost/rungekuttafehlberg78.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
#include <tatooine/linspace.h>
#include <sstream>

#ifdef Always
#undef Always
#endif
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
  grid   sample_grid{linspace{0.0, 2.0, 1000}, linspace{0.0, 1.0, 500}};
  real_t tau = 10;
  real_t t0  = 0;
  //ftle_test_vectorfield(
  //    v, sample_grid, t0, tau,
  //    "dg_ftle_" + std::to_string(t0) + "_" + std::to_string(tau));
   size_t i = 0;
   for (auto t0 : linspace{0.0, 10.0, 100}) {
    std::stringstream str;
    str << "dg.ftle." << std::setfill('0') << std::setw(2) << i++;
    ftle_test_vectorfield(v, sample_grid, t0, tau, str.str());
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
  grid<linspace<real_t>, linspace<real_t>> ftle_grid{linspace{-1.0, 1.0, 200},
                                                     linspace{-1.0, 1.0, 200}};
  auto& ftle_prop = ftle_grid.insert_scalar_vertex_property("ftle");
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
  std::seed_seq seed{100};
  //auto          tex = gpu::lic(
  //    v,
  //    uniform_grid<real_t, 2>{linspace{ftle_grid.template front<0>(),
  //                                     ftle_grid.template back<0>(), 200},
  //                            linspace{ftle_grid.template front<1>(),
  //                                     ftle_grid.template back<1>(), 100}},
  //    t0, vec<size_t, 2>{ftle_grid.size(0), ftle_grid.size(1)}, 100, 0.001,
  //    {256, 256}, seed).download_data();

  auto& ftle_prop =
      ftle_grid.scalar_vertex_property(
          "ftle");
  auto& colored_ftle_prop =
      ftle_grid.vec3_vertex_property("ftle_colored");
  ftle_field f{v, tau, ode::vclibs::rungekutta43<real_t, 2>{}};
  f.flowmap_gradient().flowmap().use_caching(false);
  f.flowmap_gradient().set_epsilon(
      vec{(ftle_grid.template back<0>() - ftle_grid.template front<0>()) /
              static_cast<real_t>(ftle_grid.size(0)),
          (ftle_grid.template back<1>() - ftle_grid.template front<1>()) /
              static_cast<real_t>(ftle_grid.size(1))});
  for_loop(
      [&](auto const ix, auto const iy) {
        vec const x{ftle_grid.vertex_at(ix, iy)};
        ftle_prop(ix, iy) = f(x, t0);
      },
      ftle_grid.size(0), ftle_grid.size(1));

  auto max = -std::numeric_limits<real_t>::max();
  ftle_grid.vertices().iterate_indices(
      [&](auto const... is) { max = std::max(ftle_prop(is...), max); });
  color_scales::viridis scale;
  for_loop(
      [&](auto const ix, auto const iy) {
        vec const x{ftle_grid.vertex_at(ix, iy)};
        colored_ftle_prop(ix, iy) =
            scale(ftle_prop(ix, iy) / max)// * 0.4 +
            //tex[ix * 4 + iy * ftle_grid.size(0) * 4] * 0.6
            ;
      },
      ftle_grid.size(0), ftle_grid.size(1));
  colored_ftle_prop.write_png(path + ".png");
}
//==============================================================================
}  // namespace tatooine::test
//==============================================================================
