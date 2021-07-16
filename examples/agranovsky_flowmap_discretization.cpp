#include <tatooine/agranovsky_flowmap_discretization.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/real.h>
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main() -> int {
  auto       v         = analytical::fields::numerical::doublegyre{};
  auto       phi       = flowmap(v);
  phi.use_caching(false);
  auto const t0        = real_t(0);
  auto const t1        = real_t(20);
  auto const delta_t   = real_t(0.1);
  auto       phi_naive = naive_flowmap_discretization<real_t, 2>{
      phi, t0, t1, vec2{0, 0}, vec2{2, 1}, 100, 50};
  auto phi_agranovsky = agranovsky_flowmap_discretization<real_t, 2>{
      phi, t0, t1, delta_t, vec2{0, 0}, vec2{2, 1}, 100, 50};
  size_t cnt = 0;
  phi_naive.forward_grid().write_vtk(
      "doublegyre_naive_flowmap_discretization_forward.vtk");
  phi_naive.backward_grid().write_vtk(
      "doublegyre_naive_flowmap_discretization_backward.vtk");
  for (auto const& step : phi_agranovsky.stepped_flowmap_discretizations()) {
    step.forward_grid().write_vtk(
        "doublegyre_agranoksy_flowmap_discretization_forward" +
        std::to_string(cnt) + ".vtk");
    step.backward_grid().write_vtk(
        "doublegyre_agranoksy_flowmap_discretization_backward" +
        std::to_string(cnt) + ".vtk");
    ++cnt;
  }

  auto g  = grid{linspace{0.0, 2.0, 1001}, linspace{0.0, 1.0, 501}};
  auto gv = g.vertices();
  // forward check
  size_t num_agranovksy_better = 0;
  size_t num_naive_better = 0;
  size_t num_about_same = 0;
  gv.iterate_indices([&](auto const... is) {
    auto const x0              = gv(is...);
    auto const x1_integrated   = phi(x0, t0, t1 - t0);
    auto const x1_naive        = phi_naive.sample_forward(x0);
    auto const x1_agranovsky   = phi_agranovsky.sample_forward(x0);
    auto const dist_naive      = distance(x1_integrated, x1_naive);
    auto const dist_agranovsky = distance(x1_integrated, x1_agranovsky);
    if (dist_naive > dist_agranovsky) {
      ++num_agranovksy_better;
    } else {
      if (dist_agranovsky - dist_naive > 1e-4) {
        ++num_naive_better;
      } else {
        ++num_about_same;
      }
    }
  });
  std::cout << num_agranovksy_better << '\n';
  std::cout << num_naive_better << '\n';
  std::cout << num_about_same << '\n';
  // backward check
  //gv.iterate_indices([&](auto const... is) {
  //    try{
  //  auto const x0              = gv(is...);
  //  auto const x1_integrated   = phi(x0, t1, t0 - t1);
  //  auto const x1_naive        = phi_naive.sample_backward(x0);
  //  auto const x1_agranovsky   = phi_agranovsky.sample_backward(x0);
  //  auto const dist_naive      = distance(x1_integrated, x1_naive);
  //  auto const dist_agranovsky = distance(x1_integrated, x1_agranovsky);
  //  if (dist_naive < dist_agranovsky) {
  //    std::cerr << "==================\n";
  //    std::cerr << "something is wrong\n";
  //    std::cerr << x0 << '\n';
  //    std::cerr << x1_integrated << '\n';
  //    std::cerr << x1_naive << '\n';
  //    std::cerr << dist_naive << '\n';
  //    std::cerr << x1_agranovsky << '\n';
  //    std::cerr << dist_agranovsky << '\n';
  //  }
  //  } catch(...){}
  //});
}
