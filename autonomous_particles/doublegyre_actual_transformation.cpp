#include <tatooine/analytical/fields/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/line.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>
#include <tatooine/geometry/sphere.h>

#include <iomanip>
#include <sstream>

#include "parse_args.h"
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  indeterminate_progress_bar([&](auto indicator) {
    indicator.set_text("Parsing Arguments");
    auto args_opt = parse_args<2>(argc, argv);
    if (!args_opt) {
      return;
    }
    auto args = *args_opt;
    //--------------------------------------------------------------------------
    auto v   = analytical::fields::numerical::doublegyre{};
    auto phi = flowmap(v);

    //--------------------------------------------------------------------------
    indicator.set_text("Advecting autonomous particles");
    auto const initial_part = autonomous_particle2{args.x0, args.t0, args.r0};
    auto transformed_circle = discretize(geometry::circle{}, 1000);
    for (auto v : transformed_circle.vertices()) {
      transformed_circle[v] =
          initial_part.S() * transformed_circle[v] + initial_part.center();
    }
    transformed_circle.write("doublegyre_single_transformed_ellipse_0.vtk");
    for (auto v : transformed_circle.vertices()) {
      transformed_circle[v] = phi(transformed_circle[v], args.t0, args.tau);
    }
    transformed_circle.write("doublegyre_single_transformed_ellipse_1.vtk");
    auto const [advected_particles, advected_simple_particles] = [&] {
      switch (args.split_behavior) {
        case split_behavior_t::two_splits:
          return initial_part
              .advect<autonomous_particle2::split_behaviors::two_splits>(
                  phi, args.step_width, args.tau);
        default:
        case split_behavior_t::three_splits:
          return initial_part
              .advect<autonomous_particle2::split_behaviors::three_splits>(
                  phi, args.step_width, args.tau);
        // case split_behavior_t::five_splits:
        //   return initial_part
        //       .advect<autonomous_particle2::split_behaviors::five_splits>(
        //           phi, args.step_width, args.tau);
        // case split_behavior_t::seven_splits:
        //   return initial_part
        //       .advect<autonomous_particle2::split_behaviors::seven_splits>(
        //           phi, args.step_width, args.tau);
        case split_behavior_t::centered_four:
          return initial_part
              .advect<autonomous_particle2::split_behaviors::centered_four>(
                  phi, args.step_width, args.tau);
      }
    }();
    std::cerr << "number of advected particles: " << size(advected_particles)
              << '\n';
    //--------------------------------------------------------------------------
    indicator.set_text("Writing Autonomous Particles Results");
    auto all_advected_discretizations = std::vector<line2>{};
    auto all_initial_discretizations  = std::vector<line2>{};
    for (auto const& p : advected_particles) {
      all_initial_discretizations.push_back(
          discretize(p.initial_ellipse(), 33));
      all_advected_discretizations.push_back(discretize(p, 33));
    }
    all_initial_discretizations.front().write_vtk(
        "doublegyre_single_front.vtp");
    write_vtk(all_initial_discretizations, "doublegyre_single_ellipsoids0.vtk");
    write_vtk(all_advected_discretizations,
              "doublegyre_single_ellipsoids1.vtk");
  });
}
