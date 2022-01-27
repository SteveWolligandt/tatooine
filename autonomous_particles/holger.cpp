#include <tatooine/autonomous_particle.h>
#include <tatooine/analytical/fields/doublegyre.h>

#include <vector>
//==============================================================================
using namespace tatooine;
using namespace detail::autonomous_particle;
//==============================================================================
auto doublegyre_example() {
  auto v              = analytical::fields::numerical::doublegyre{};
  auto uuid_generator = std::atomic_uint64_t{};
  auto initial_grid = rectilinear_grid{linspace{0.8, 1.2, 5},
                                       linspace{0.4, 0.6, 3}};
  auto const [advected_particles, advected_simple_particles, edges] =
      autonomous_particle2::advect_with_three_splits(flowmap(v), 0.01, 0, 3,
                                                     initial_grid);

  auto advected_discretizations        = std::vector<line2>{};
  auto initial_discretizations         = std::vector<line2>{};
  auto advected_circle_discretizations = std::vector<line2>{};
  auto initial_circle_discretizations  = std::vector<line2>{};
  advected_discretizations.reserve(size(advected_particles));
  initial_discretizations.reserve(size(advected_particles));
  advected_circle_discretizations.reserve(size(advected_particles));
  initial_circle_discretizations.reserve(size(advected_particles));

  auto discretized = [](auto const& ell) {
    return discretize(ell, 64);
  };
  auto as_initial  = [](auto const& p) {
    return p.initial_ellipse();
  };
  auto advected_circle_discretized = [](auto const& p) {
    auto ell = p.initial_ellipse();
    auto const [axes, lengths] = ell.main_axes();
    if (std::abs(lengths(0) - lengths(1)) > 1e-10) {
      ell.S() = axes * diag(vec{lengths(0), lengths(0)}) * transposed(axes);
      //ell.S() = mat2::eye() * diag(vec{lengths(0), lengths(0)});
    }
    ell.S()      = ell.S() * p.nabla_phi();
    ell.center() = p.center();
    return discretize(ell, 64);
  };
  auto initial_circle_discretized  = [](auto const& p) {
    auto ell = p.initial_ellipse();
    auto const [axes, lengths] = ell.main_axes();
    ell.S() = axes * diag(vec{lengths(0), lengths(0)}) * transposed(axes);
    return discretize(ell, 64);
  };

  using namespace std::ranges;
  copy(advected_particles
       | views::transform(discretized),
       std::back_inserter(advected_discretizations));
  copy(advected_particles
       | views::transform(as_initial)
       | views::transform(discretized),
       std::back_inserter(initial_discretizations));
  copy(advected_particles
       | views::transform(advected_circle_discretized),
       std::back_inserter(advected_circle_discretizations));
  copy(advected_particles
       | views::transform(initial_circle_discretized),
       std::back_inserter(initial_circle_discretizations));

  auto centers = pointset2{};
  auto& dir0  = centers.vec2_vertex_property("dir0");
  auto& dir1  = centers.vec2_vertex_property("dir1");
  for (auto const& p : advected_particles) {
    auto v = centers.insert_vertex(p.center());
    dir0[v] = p.nabla_phi().col(0);
    dir1[v] = p.nabla_phi().col(1);
  }

  write(advected_discretizations, "holger_doublegyre_ellipses1.vtk");
  write(initial_discretizations, "holger_doublegyre_ellipses0.vtk");
  write(advected_circle_discretizations,
        "holger_doublegyre_ellipses1_circle.vtk");
  write(initial_circle_discretizations,
        "holger_doublegyre_ellipses0_circle.vtk");
  centers.write("holger_directions.vtp");
}
//==============================================================================
auto main() -> int {
  doublegyre_example();
  // artificial_example();
}
