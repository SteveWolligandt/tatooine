#include <tatooine/analytical/modified_doublegyre.h>
#include <tatooine/linspace.h>
#include <tatooine/rendering/interactive.h>
#include <tatooine/numerical_flowmap.h>
//==============================================================================
using namespace tatooine;
using analytical::numerical::modified_doublegyre;
//==============================================================================
auto main() -> int {
  auto const v = modified_doublegyre{};
  auto phi = flowmap(v);
  auto ht_top = line3{};
  auto ht_bottom = line3{};
  auto ht_bottom_ground_truth = line3{};
  auto ht_top_ground_truth = line3{};
  auto const x0 = v.hyperbolic_trajectory(0);
  for (auto const tau : linspace{0.0, 10.0, 100}) {
    auto const x1              = phi(vec{x0.x(), 1}, 0, tau);
    auto const x1_ground_truth = v.hyperbolic_trajectory(tau);
    ht_bottom_ground_truth.push_back(vec{x1.x(), 0, tau});
    ht_bottom.push_back(vec{x1.x(), 0, tau});
    ht_top.push_back(vec{x1.x(), 1, tau});
    ht_top_ground_truth.push_back(vec{x1.x(), 1, tau});
  }

  auto lcs = v.lagrangian_coherent_structure(0);
  auto discretized_lcs = line2{};

  for (auto const t : linspace{0.0, 30.0, 1000}) {
    discretized_lcs.push_back(lcs(t));
  }
  discretized_lcs.write("modified_doublegyre.lcs.vtp");
  ht_top_ground_truth.write("modified_doublegyre.ht_top_ground_truth.vtp");
  ht_bottom_ground_truth.write(
      "modified_doublegyre.ht_bottom_ground_truth.vtp");
  ht_top.write("modified_doublegyre.ht_top.vtp");
  ht_bottom.write("modified_doublegyre.ht_bottom.vtp");
}
