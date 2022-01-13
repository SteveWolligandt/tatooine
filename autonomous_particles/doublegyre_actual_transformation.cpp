#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/geometry/sphere.h>
#include <tatooine/line.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>

#include <iomanip>
#include <sstream>

#include "parse_args.h"
//==============================================================================
namespace tatooine::autonomous_particles {
auto doublegyre_actual_transformation(args_t const&) -> void;
}
//==============================================================================
auto main(int argc, char** argv) -> int {
  using namespace tatooine::autonomous_particles;
  auto args_opt = parse_args<2>(argc, argv);
  if (!args_opt) {
    return;
  }
  auto args = *args_opt;
  doublegyre_actual_transformation(args);
}
//==============================================================================
namespace tatooine::autonomous_particles {
//==============================================================================
auto doublegyre_actual_transformation(args_t const& args) -> void {
  indeterminate_progress_bar([&](auto indicator) {
    indicator.set_text("Parsing Arguments");
    //--------------------------------------------------------------------------
    auto v   = analytical::fields::numerical::doublegyre{};
    auto phi = flowmap(v);
    //--------------------------------------------------------------------------
    indicator.set_text("Advecting autonomous particles");
    auto particles =
        std::vector{autonomous_particle2{args.x0, args.t0, args.r0}};
    {
      auto discretizations = std::vector<line2>{};
      for (auto const& p : particles) {
        discretizations.push_back(discretize(p, 33));
      }
      write(discretizations, "doublegyre_single_ellipsoids0.vtk");
    }
    auto transformed_circle = discretize(geometry::circle{}, 1000);
    for (auto v : transformed_circle.vertices()) {
      transformed_circle[v] = particles.front().S() * transformed_circle[v] +
                              particles.front().center();
    }
    transformed_circle.write("doublegyre_single_transformed_ellipse0.vtk");
    auto const  stepwidth = 0.1;
    auto        t         = args.t0;
    std::size_t i         = 1;

    while (t < args.t0 + args.tau) {
      particles = std::get<0>(advect(particles, args.split_behavior));
      //--------------------------------------------------------------------------
      indicator.set_text("Writing Autonomous Particles Results");
      using namespace std::ranges;
      if (!particles.empty()) {
        auto discretizations = std::vector<line2>{};
        discretizations.reserve(particles.size());
        auto discretize33 = [](auto const& p) { return discretize(p, 33); };
        copy(particles | views::transform(discretize33),
             std::back_inserter(discretizations));
        write(discretizations,
              "doublegyre_single_ellipsoids" + std::to_string(i) + ".vtk");
      }
      for (auto v : transformed_circle.vertices()) {
        transformed_circle[v] = phi(transformed_circle[v], t, stepwidth);
      }
      transformed_circle.write("doublegyre_single_transformed_ellipse" +
                               std::to_string(i) + ".vtk");
      t += stepwidth;
      ++i;
    }
  });
}
//==============================================================================
}  // namespace tatooine::autonomous_particles
//==============================================================================
