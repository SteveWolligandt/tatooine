#include <tatooine/agranovsky_flowmap_discretization.h>
#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/autonomous_particle_flowmap_discretization.h>
#include <tatooine/chrono.h>
#include <tatooine/naive_flowmap_discretization.h>
#include <tatooine/netcdf.h>
#include <tatooine/progress_bars.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/unstructured_triangular_grid.h>
#include <tatooine/vtk_legacy.h>

#include <iomanip>
#include <sstream>

#include "parse_args.h"
#include "write_ellipses.h"
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  std::stringstream report;
  report << "==============================================================="
            "=================\n"
         << "REPORT\n"
         << "==============================================================="
            "=================\n";
  auto args_opt = parse_args(argc, argv);
  if (!args_opt) {
    return 1;
  }
  auto args = *args_opt;
  report << "t0: " << args.t0 << '\n' << "tau: " << args.tau << '\n';

  auto v = analytical::fields::numerical::doublegyre{};
  v.set_infinite_domain(true);

  indeterminate_progress_bar([&](auto indicator) {
    //----------------------------------------------------------------------------
    indicator.set_text("Building numerical flowmap");
    auto phi = flowmap(v);
    phi.use_caching(false);
    //----------------------------------------------------------------------------
    indicator.set_text("Discretizing flow map with autonomous particles");
    auto autonomous_disc = autonomous_particle_flowmap_discretization_2{
        phi, args.t0, args.tau, args.tau_step,
        rectilinear_grid{linspace{0.0, 2.0, args.width + 1},
                         linspace{0.0, 1.0, args.height + 1}}};
    auto const num_particles_after_advection = size(autonomous_disc.samplers());
    //----------------------------------------------------------------------------
    indicator.set_text("Writing Autonomous Particles Results");
    {
      autonomous_disc.mesh0().write_vtk("doublegyre_grid_autonomous_mesh0.vtk");
      autonomous_disc.mesh1().write_vtk("doublegyre_grid_autonomous_mesh1.vtk");
      std::vector<line2> all_advected_discretizations;
      std::vector<line2> all_initial_discretizations;
      for (auto const& sampler : autonomous_disc.samplers()) {
        all_initial_discretizations.push_back(
            discretize(sampler.ellipse0(), 100));
        all_advected_discretizations.push_back(
            discretize(sampler.ellipse1(), 100));
      }
      write_vtk(all_initial_discretizations, "doublegyre_grid_ellipses0.vtk");
      write_vtk(all_advected_discretizations, "doublegyre_grid_ellipses1.vtk");
    }
    //----------------------------------------------------------------------------
    indicator.set_text("Discretizing flow map naively");
    auto const regularized_height = static_cast<size_t>(
        std::ceil(std::sqrt(num_particles_after_advection / 2)));

    auto naive_disc =
        naive_flowmap_discretization<real_t, 2>{phi,
                                                args.t0,
                                                args.tau,
                                                vec2{0, 0},
                                                vec2{2, 1},
                                                regularized_height * 2,
                                                regularized_height};
    //----------------------------------------------------------------------------
    indicator.set_text("Discretizing flow map with agranovsky sampling");
    real_t const agranovsky_delta_t = 0.1;
    auto         agranovsky_disc =
        AgranovskyFlowmapDiscretization<2>{phi,
                                           args.t0,
                                           args.tau,
                                           agranovsky_delta_t,
                                           vec2{0, 0},
                                           vec2{2, 1},
                                           regularized_height * 2,
                                           regularized_height};
    {
      size_t i = 0;
      for (auto const& step : agranovsky_disc.steps()) {
        step.backward_grid().write_vtk("agranovsky_backward_" +
                                       std::to_string(i++) + ".vtk");
      }
    }

    //----------------------------------------------------------------------------
    indicator.set_text(
        "Discretizing flow map with staggered autonomous particles");
    real_t const autonomous_particles_delta_t = 1;
    auto         staggered_autonomous_disc =
        staggered_autonomous_particle_flowmap_discretization_2{
            phi,
            args.t0,
            args.tau,
            autonomous_particles_delta_t,
            args.tau_step,
            rectilinear_grid{linspace{0.0, 2.0, args.width + 1},
                             linspace{0.0, 1.0, args.height + 1}}};

    // indicator.set_text("Writing ellipses to NetCDF");
    // if (args.write_ellipses_to_netcdf) {
    //  write_x1(initial_particles, "doublegyre_grid_initial.nc");
    //  write_x1(autonomous_particles, "doublegyre_grid_advected.nc");
    //  write_x0(autonomous_particles, "doublegyre_grid_back_calculation.nc");
    //}
    //----------------------------------------------------------------------------
    // Create memory for measuring
    //----------------------------------------------------------------------------
    rectilinear_grid sampler_check_grid{linspace{0.0, 2.0, args.output_res_x},
                                        linspace{0.0, 1.0, args.output_res_y}};
    [[maybe_unused]] auto& autonomous_flowmap_forward_prop =
        sampler_check_grid.vec2_vertex_property("autonomous_flowmap_forward");
    [[maybe_unused]] auto& autonomous_flowmap_backward_prop =
        sampler_check_grid.vec2_vertex_property("autonomous_flowmap_backward");
    [[maybe_unused]] auto& staggered_autonomous_flowmap_forward_prop =
        sampler_check_grid.vec2_vertex_property(
            "staggered_autonomous_flowmap_forward");
    [[maybe_unused]] auto& staggered_autonomous_flowmap_backward_prop =
        sampler_check_grid.vec2_vertex_property(
            "staggered_autonomous_flowmap_backward");
    [[maybe_unused]] auto& forward_errors_autonomous_prop =
        sampler_check_grid.scalar_vertex_property("forward_error_autonomous");
    [[maybe_unused]] auto& forward_errors_staggered_autonomous_prop =
        sampler_check_grid.scalar_vertex_property(
            "forward_error_staggered_autonomous");
    [[maybe_unused]] auto& forward_errors_naive_prop =
        sampler_check_grid.scalar_vertex_property("forward_error_naive");
    [[maybe_unused]] auto& forward_errors_agranovsky_prop =
        sampler_check_grid.scalar_vertex_property("forward_error_agranovsky");
    [[maybe_unused]] auto& forward_errors_diff_naive_prop =
        sampler_check_grid.scalar_vertex_property("forward_error_diff_naive");
    [[maybe_unused]] auto& forward_errors_diff_agranovsky_prop =
        sampler_check_grid.scalar_vertex_property(
            "forward_error_diff_agranovsky");
    [[maybe_unused]] auto& forward_errors_diff_staggered_prop =
        sampler_check_grid.scalar_vertex_property(
            "forward_error_diff_staggered");
    [[maybe_unused]] auto& backward_errors_autonomous_prop =
        sampler_check_grid.scalar_vertex_property("backward_error_autonomous");
    [[maybe_unused]] auto& backward_errors_staggered_autonomous_prop =
        sampler_check_grid.scalar_vertex_property(
            "backward_error_staggered_autonomous");
    [[maybe_unused]] auto& backward_errors_naive_prop =
        sampler_check_grid.scalar_vertex_property("backward_error_naive");
    [[maybe_unused]] auto& backward_errors_agranovsky_prop =
        sampler_check_grid.scalar_vertex_property("backward_error_agranovsky");
    [[maybe_unused]] auto& backward_errors_diff_naive_prop =
        sampler_check_grid.scalar_vertex_property("backward_error_diff_naive");
    [[maybe_unused]] auto& backward_errors_diff_agranovsky_prop =
        sampler_check_grid.scalar_vertex_property(
            "backward_error_diff_agranovsky");
    [[maybe_unused]] auto& backward_errors_diff_staggered_prop =
        sampler_check_grid.scalar_vertex_property(
            "backward_error_diff_staggered");

    real_t mean_autonomous_forward_error = std::numeric_limits<real_t>::max(),
           mean_staggered_autonomous_forward_error =
               std::numeric_limits<real_t>::max(),
           mean_naive_forward_error       = std::numeric_limits<real_t>::max(),
           mean_agranovsky_forward_error  = std::numeric_limits<real_t>::max();
    real_t mean_autonomous_backward_error = std::numeric_limits<real_t>::max(),
           mean_staggered_autonomous_backward_error =
               std::numeric_limits<real_t>::max(),
           mean_naive_backward_error      = std::numeric_limits<real_t>::max(),
           mean_agranovsky_backward_error = std::numeric_limits<real_t>::max();
    size_t num_points_ood_forward = 0, num_points_ood_backward = 0;
    std::mutex error_mutex;

    //----------------------------------------------------------------------------
    // Compare forward flow map
    //----------------------------------------------------------------------------
    indicator.set_text("Comparing forward flow map");
    {
      std::vector<real_t> autonomous_errors, staggered_autonomous_errors,
          naive_errors, agranovsky_errors;
      autonomous_errors.reserve(sampler_check_grid.vertices().size());
      staggered_autonomous_errors.reserve(sampler_check_grid.vertices().size());
      naive_errors.reserve(sampler_check_grid.vertices().size());
      agranovsky_errors.reserve(sampler_check_grid.vertices().size());
      sampler_check_grid.vertices().iterate_indices(
          [&](auto const... is) {
            auto const x0           = sampler_check_grid.vertex_at(is...);
            auto const numerical_x1 = phi(x0, args.t0, args.tau);
            try {
              auto const x1 = autonomous_disc.sample_forward(x0);
              autonomous_flowmap_forward_prop(is...) = x1;

              {std::lock_guard lock{error_mutex};
              autonomous_errors.push_back(distance(x1, numerical_x1));
              forward_errors_autonomous_prop(is...) = autonomous_errors.back();}

            } catch (std::exception const& e) {
              autonomous_flowmap_forward_prop(is...) = vec2::ones() * 0.0 / 0.0;
              forward_errors_autonomous_prop(is...)  = 0.0 / 0.0;
            }
            try {
              auto const x1 = staggered_autonomous_disc.sample_forward(x0);
              staggered_autonomous_flowmap_forward_prop(is...) = x1;

              {std::lock_guard lock{error_mutex};
              staggered_autonomous_errors.push_back(distance(x1, numerical_x1));
              forward_errors_staggered_autonomous_prop(is...) =
                  staggered_autonomous_errors.back();}

            } catch (std::exception const& e) {
              staggered_autonomous_flowmap_forward_prop(is...) =
                  vec2::ones() * 0.0 / 0.0;
              forward_errors_staggered_autonomous_prop(is...) = 0.0 / 0.0;
            }
            try {
              auto const x1 = naive_disc.sample_forward(x0);

              {std::lock_guard lock{error_mutex};
              naive_errors.push_back(distance(x1, numerical_x1));
              forward_errors_naive_prop(is...) = naive_errors.back();}

            } catch (std::exception const& e) {
              forward_errors_naive_prop(is...) = 0.0 / 0.0;
            }
            try {
              auto const x1 = agranovsky_disc.sample_forward(x0);

              {std::lock_guard lock{error_mutex};
              agranovsky_errors.push_back(distance(x1, numerical_x1));
              forward_errors_agranovsky_prop(is...) = agranovsky_errors.back();}

            } catch (std::exception const& e) {
              forward_errors_agranovsky_prop(is...) = 0.0 / 0.0;
            }
            forward_errors_diff_naive_prop(is...) =
                forward_errors_naive_prop(is...) -
                forward_errors_autonomous_prop(is...);
            forward_errors_diff_agranovsky_prop(is...) =
                forward_errors_agranovsky_prop(is...) -
                forward_errors_autonomous_prop(is...);
            forward_errors_diff_agranovsky_prop(is...) =
                forward_errors_staggered_autonomous_prop(is...) -
                forward_errors_autonomous_prop(is...);
          },
          tag::parallel);
      mean_autonomous_forward_error =
          std::accumulate(begin(autonomous_errors), end(autonomous_errors),
                          real_t(0)) /
          size(autonomous_errors);
      mean_staggered_autonomous_forward_error =
          std::accumulate(begin(staggered_autonomous_errors),
                          end(staggered_autonomous_errors), real_t(0)) /
          size(staggered_autonomous_errors);
      mean_naive_forward_error =
          std::accumulate(begin(naive_errors), end(naive_errors), real_t(0)) /
          size(naive_errors);
      mean_agranovsky_forward_error =
          std::accumulate(begin(agranovsky_errors), end(agranovsky_errors),
                          real_t(0)) /
          size(agranovsky_errors);
    }
    //----------------------------------------------------------------------------
    // Compare backward flow map
    //----------------------------------------------------------------------------
    indicator.set_text("Comparing backward flow map");
    {
      std::vector<real_t> autonomous_errors, staggered_autonomous_errors,
          naive_errors, agranovsky_errors;
      autonomous_errors.reserve(sampler_check_grid.vertices().size());
      naive_errors.reserve(sampler_check_grid.vertices().size());
      agranovsky_errors.reserve(sampler_check_grid.vertices().size());
      sampler_check_grid.vertices().iterate_indices(
          [&](auto const... is) {
            auto const x1           = sampler_check_grid.vertex_at(is...);
            auto const numerical_x0 = [&] {
              static std::mutex m;
              std::lock_guard   l{m};
              return phi(x1, args.t0 + args.tau, -args.tau);
            }();
            try {
              auto const x0 = autonomous_disc.sample_backward(x1);
              autonomous_flowmap_backward_prop(is...) = x0;

              {std::lock_guard lock{error_mutex};
              autonomous_errors.push_back(distance(x0, numerical_x0));
              backward_errors_autonomous_prop(is...) = autonomous_errors.back();}

            } catch (std::exception const& e) {
              autonomous_flowmap_backward_prop(is...) =
                  vec2::ones() * 0.0 / 0.0;
              backward_errors_autonomous_prop(is...) = 0.0 / 0.0;
            }
            try {
              auto const x0 = staggered_autonomous_disc.sample_backward(x1);
              staggered_autonomous_flowmap_backward_prop(is...) = x0;

              {std::lock_guard lock{error_mutex};
              staggered_autonomous_errors.push_back(distance(x0, numerical_x0));
              backward_errors_staggered_autonomous_prop(is...) =
                staggered_autonomous_errors.back();}

            } catch (std::exception const& e) {
              staggered_autonomous_flowmap_backward_prop(is...) =
                  vec2::ones() * 0.0 / 0.0;
              backward_errors_staggered_autonomous_prop(is...) = 0.0 / 0.0;
            }
            try {
              auto const x0 = naive_disc.sample_backward(x1);

              {std::lock_guard lock{error_mutex};
              naive_errors.push_back(distance(x0, numerical_x0));
              backward_errors_naive_prop(is...) = naive_errors.back();}

            } catch (std::exception const& e) {
              backward_errors_naive_prop(is...) = 0.0 / 0.0;
            }
            try {
              auto const x0 = agranovsky_disc.sample_backward(x1);

              {std::lock_guard lock{error_mutex};
              agranovsky_errors.push_back(distance(x0, numerical_x0));
              backward_errors_agranovsky_prop(is...) = agranovsky_errors.back();}

            } catch (std::exception const& e) {
              backward_errors_agranovsky_prop(is...) = 0.0 / 0.0;
            }
            backward_errors_diff_naive_prop(is...) =
                backward_errors_naive_prop(is...) -
                backward_errors_autonomous_prop(is...);
            backward_errors_diff_agranovsky_prop(is...) =
                backward_errors_agranovsky_prop(is...) -
                backward_errors_autonomous_prop(is...);
            backward_errors_diff_agranovsky_prop(is...) =
                backward_errors_staggered_autonomous_prop(is...) -
                backward_errors_autonomous_prop(is...);
          },
          tag::parallel);
      mean_autonomous_backward_error =
          std::accumulate(begin(autonomous_errors), end(autonomous_errors),
                          real_t(0)) /
          size(autonomous_errors);
      mean_staggered_autonomous_backward_error =
          std::accumulate(begin(staggered_autonomous_errors),
                          end(staggered_autonomous_errors), real_t(0)) /
          size(staggered_autonomous_errors);
      mean_naive_backward_error =
          std::accumulate(begin(naive_errors), end(naive_errors), real_t(0)) /
          size(naive_errors);
      mean_agranovsky_backward_error =
          std::accumulate(begin(agranovsky_errors), end(agranovsky_errors),
                          real_t(0)) /
          size(agranovsky_errors);
    }
    //----------------------------------------------------------------------------
    indicator.set_text("Writing results");
    { sampler_check_grid.write("doublegyre_grid_errors.vtk"); }
    //----------------------------------------------------------------------------
    indicator.mark_as_completed();
    report << num_points_ood_forward << " / "
           << sampler_check_grid.vertices().size()
           << " out of domain in forward direction("
           << (100 * num_points_ood_forward /
               (real_t)sampler_check_grid.vertices().size())
           << "%)\n"
           << num_points_ood_backward << " / "
           << sampler_check_grid.vertices().size()
           << " out of domain in backward direction("
           << (100 * num_points_ood_backward /
               (real_t)sampler_check_grid.vertices().size())
           << "%)\n"
           << "mean error forward autonomous particles: " << std::scientific
           << mean_autonomous_forward_error << '\n'
           << "mean error forward staggered autonomous particles: " << std::scientific
           << mean_staggered_autonomous_forward_error << '\n'
           << "mean error forward naive grid: " << std::scientific
           << mean_naive_forward_error << '\n'
           << "mean error forward agranovsky grid: " << std::scientific
           << mean_agranovsky_forward_error << '\n'
           << "mean error backward autonomous particles: " << std::scientific
           << mean_autonomous_backward_error << '\n'
           << "mean error backward staggered autonomous particles: " << std::scientific
           << mean_staggered_autonomous_backward_error << '\n'
           << "mean error backward naive grid: " << std::scientific
           << mean_naive_backward_error << '\n'
           << "mean error backward agranovsky grid: " << std::scientific
           << mean_agranovsky_backward_error << '\n';

    if (mean_naive_forward_error > mean_autonomous_forward_error &&
        mean_agranovsky_forward_error > mean_autonomous_forward_error &&
        mean_staggered_autonomous_forward_error > mean_autonomous_forward_error) {
      report << "autonomous particles are better in forward direction\n";
    } else if (mean_agranovsky_forward_error > mean_naive_forward_error &&
               mean_autonomous_forward_error > mean_naive_forward_error &&
               mean_staggered_autonomous_forward_error > mean_naive_forward_error) {
      report << "naive grid is better in forward direction\n";
    } else if (mean_naive_forward_error > mean_agranovsky_forward_error &&
               mean_autonomous_forward_error > mean_agranovsky_forward_error &&
               mean_staggered_autonomous_forward_error > mean_agranovsky_forward_error) {
      report << "agranovsky is better in forward direction\n";
    } else if (mean_naive_forward_error > mean_staggered_autonomous_forward_error &&
               mean_autonomous_forward_error > mean_staggered_autonomous_forward_error &&
               mean_agranovsky_forward_error > mean_staggered_autonomous_forward_error) {
      report << "agranovsky is better in forward direction\n";
    }

    if (mean_naive_backward_error > mean_autonomous_backward_error &&
        mean_agranovsky_backward_error > mean_autonomous_backward_error &&
        mean_staggered_autonomous_backward_error > mean_autonomous_backward_error) {
      report << "autonomous particles are better in backward direction\n";
    } else if (mean_agranovsky_backward_error > mean_naive_backward_error &&
               mean_autonomous_backward_error > mean_naive_backward_error &&
               mean_staggered_autonomous_backward_error > mean_naive_backward_error) {
      report << "naive grid is better in backward direction\n";
    } else if (mean_naive_backward_error > mean_agranovsky_backward_error &&
               mean_autonomous_backward_error > mean_agranovsky_backward_error &&
               mean_staggered_autonomous_backward_error > mean_agranovsky_backward_error) {
      report << "agranovsky is better in backward direction\n";
    } else if (mean_naive_backward_error > mean_staggered_autonomous_backward_error &&
               mean_autonomous_backward_error > mean_staggered_autonomous_backward_error &&
               mean_agranovsky_backward_error > mean_staggered_autonomous_backward_error) {
      report << "agranovsky is better in backward direction\n";
    }
  });
  std::cerr << report.str();
}
