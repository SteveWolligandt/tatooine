#include <tatooine/analytical/numerical/counterexample_sadlo.h>
#include <tatooine/analytical/numerical/doublegyre.h>
#include <tatooine/analytical/numerical/modified_doublegyre.h>
#include <tatooine/analytical/numerical/saddle.h>
#include <tatooine/color_scales/viridis.h>
#include <tatooine/for_loop.h>
#include <tatooine/ftle_field.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/gpu/lic.h>
#include <tatooine/linspace.h>
#include <sstream>

//==============================================================================
namespace tatooine::examples {
//==============================================================================
using namespace tatooine::analytical::numerical;
template <typename V, typename Grid, typename T0, typename Tau>
auto ftle_test_vectorfield(V const&, Grid&, T0, Tau, std::string const&) -> void;
//==============================================================================
auto ftle_doublegyre() {
  auto v = doublegyre{};
  v.set_infinite_domain(true);
  auto sample_grid =
      rectilinear_grid{linspace{0.0, 2.0, 100}, linspace{0.0, 1.0, 50}};
  auto const       tau = real_number{10};
  auto             i   = std::size_t{};
  for (auto t0 : linspace{0.0, 10.0, 10}) {
    auto str = std::stringstream{};
    str << "dg.ftle." << std::setfill('0') << std::setw(2) << i++;
    ftle_test_vectorfield(v, sample_grid, t0, tau, str.str());
  }
}
//==============================================================================
auto ftle_modified_doublegyre() {
  auto v = modified_doublegyre{};
  auto sample_grid =
      rectilinear_grid{linspace{0.0, 2.0, 100}, linspace{-1.0, 1.0, 50}};
  for (auto t0 : linspace(1.0, 3.0, 3)) {
    for (auto tau : std::array{-20, 20}) {
      ftle_test_vectorfield(
          v, sample_grid, t0, tau,
          "ndg_ftle_" + std::to_string(tau) + "_" + std::to_string(t0));
    }
  }
}
//==============================================================================
auto ftle_counterexample_sadlo() {
  auto v = counterexample_sadlo{};
  auto sample_grid =
      rectilinear_grid{linspace{-3.0, 3.0, 100}, linspace{-3.0, 3.0, 100}};
  ftle_test_vectorfield(v, sample_grid, 0.0, 10.0,
                        "ftle_counterexample_sadlo_ftle_tau_" +
                            std::to_string(10.0) + "_t0_" +
                            std::to_string(0.0));
}
//==============================================================================
auto ftle_saddle() {
  auto v = saddle{};
  auto domain =
      rectilinear_grid<linspace<real_number>, linspace<real_number>>{
          linspace{-1.0, 1.0, 100}, linspace{-1.0, 1.0, 100}};
  auto&      ftle_prop = domain.insert_scalar_vertex_property("ftle");
  auto const t0        = 0;
  auto const tau       = 10;

  auto f = ftle_field{v, tau};
  for_loop(
      [&](auto const... is) {
        ftle_prop(is...) = f(vec{domain.vertex_at(is...)}, t0);
      },
      200, 200);
  domain.write_vtk("ftle_saddle_tau" + std::to_string(tau) + "_t0_" +
                      std::to_string(t0) + ".vtk");
}
//==============================================================================
template <typename V, typename Grid, typename T0, typename Tau>
auto ftle_test_vectorfield(V const& v, Grid& domain, T0 t0, Tau tau,
                           std::string const& path) -> void {
  auto seed = std::seed_seq{100};

  auto f = ftle_field{v, tau};
  f.flowmap_gradient().flowmap().use_caching(false);
  f.flowmap_gradient().set_epsilon(
      vec{(domain.template back<0>() - domain.template front<0>()) /
              static_cast<real_number>(domain.size(0)),
          (domain.template back<1>() - domain.template front<1>()) /
              static_cast<real_number>(domain.size(1))});
  auto  max       = 1.2;
  auto  scale     = color_scales::viridis{};
  auto& ftle_prop = domain.sample_to_vertex_property(
      [&](auto const& p) { return scale(std::max(max, f(p, t0)) / max); },
      "ftle");

  ftle_prop.write_png(path + ".png");
}
//==============================================================================
}  // namespace tatooine::examples
//==============================================================================
auto main() -> int {
  tatooine::examples::ftle_doublegyre();
  tatooine::examples::ftle_modified_doublegyre();
  tatooine::examples::ftle_counterexample_sadlo();
  tatooine::examples::ftle_saddle();
}
