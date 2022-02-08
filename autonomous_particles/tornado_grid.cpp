#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/flowmap_agranovsky.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/netcdf.h>
#include <tatooine/progress_bars.h>
#include <tatooine/unstructured_tetrahedral_grid.h>
#include <tatooine/vtk_legacy.h>

#include <iomanip>
#include <sstream>

#include "parse_args.h"
#include "write_ellipses.h"
//==============================================================================
using namespace tatooine;
//==============================================================================
double const bad_threshold               = 1e-5;
double const really_bad_threshold        = 1e-4;
double const really_really_bad_threshold = 1e-3;
//==============================================================================
auto create_initial_distribution(args_t const& args) {
  rectilinear_grid initial_distribution_grid{linspace{-1.0, 1.0, args.width + 1},
                                 linspace{-1.0, 1.0, args.height + 1},
                                 linspace{-1.0, 1.0, args.depth + 1}};
  initial_distribution_grid.dimension<0>().pop_front();
  initial_distribution_grid.dimension<1>().pop_front();
  auto const spacing_x = initial_distribution_grid.dimension<0>().spacing();
  auto const spacing_y = initial_distribution_grid.dimension<1>().spacing();
  initial_distribution_grid.dimension<0>().front() -= spacing_x / 2;
  initial_distribution_grid.dimension<0>().back() -= spacing_x / 2;
  initial_distribution_grid.dimension<1>().front() -= spacing_y / 2;
  initial_distribution_grid.dimension<1>().back() -= spacing_y / 2;
  return initial_distribution_grid;
}
//------------------------------------------------------------------------------
auto create_initial_particles(args_t const& args) {
  auto       initial_distribution_grid = create_initial_distribution(args);
  auto const r0 = initial_distribution_grid.dimension<0>().spacing() / 2;
  typename autonomous_particle<real_type, 3>::container_t
      initial_particles;
  for (auto const& x : initial_distribution_grid.vertices()) {
    initial_particles.emplace_back(x, args.t0, r0);
  }

  // overlapping particles
  initial_distribution_grid.front<0>() +=
      initial_distribution_grid.dimension<0>().spacing() / 2;
  initial_distribution_grid.back<0>() +=
      initial_distribution_grid.dimension<0>().spacing() / 2;
  initial_distribution_grid.front<1>() +=
      initial_distribution_grid.dimension<1>().spacing() / 2;
  initial_distribution_grid.back<1>() +=
      initial_distribution_grid.dimension<1>().spacing() / 2;
  initial_distribution_grid.dimension<0>().pop_back();
  initial_distribution_grid.dimension<1>().pop_back();
  for (auto const& x : initial_distribution_grid.vertices()) {
    initial_particles.emplace_back(x, args.t0, r0);
  }
  
  return initial_particles;
}
//------------------------------------------------------------------------------
auto create_autonomous_mesh(range auto const& advected_particles) {
  unstructured_tetrahedral_grid<double, 3> autonomous_mesh;
  auto&                      autonomous_flowmap_mesh_prop =
      autonomous_mesh.add_vertex_property<vec3>("flowmap");
  autonomous_mesh.vertex_data().reserve(autonomous_mesh.vertices().size() +
                                        size(advected_particles));
  for (auto const& p : advected_particles) {
    auto v                          = autonomous_mesh.insert_vertex(p.x0());
    autonomous_flowmap_mesh_prop[v] = p.x1();
  }
  return autonomous_mesh;
}
//------------------------------------------------------------------------------
auto create_agranovsky_flowmap(auto&& v, size_t const grid_min_extent,
                               args_t const& args) {
  double const agranovsky_delta_t = 1;
  return flowmap_agranovsky{v,
                            args.t0,
                            args.tau,
                            agranovsky_delta_t,
                            vec3{-1, -1, -1},
                            vec3{1, 1, 1},
                            grid_min_extent * 2,
                            grid_min_extent};
}
//------------------------------------------------------------------------------
auto compare_direct_forward(auto const& advected_particles,
                            auto&& numerical_flowmap, auto&& agranovsky,
                            auto const& args, auto& report) {
  size_t num_autonomous_better_forward            = 0;
  size_t num_autonomous_bad_forward               = 0;
  size_t num_autonomous_really_bad_forward        = 0;
  size_t num_autonomous_really_really_bad_forward = 0;
  size_t num_agranovsky_bad_forward               = 0;
  size_t num_agranovsky_really_bad_forward        = 0;
  size_t num_agranovsky_really_really_bad_forward = 0;
  for (auto const& p : advected_particles) {
    auto const numerical_x0    = numerical_flowmap(p.x0(), args.t0, args.tau);
    auto const dist_autonomous = distance(numerical_x0, p.x1());
    if (dist_autonomous > bad_threshold) {
      ++num_autonomous_bad_forward;
    }
    if (dist_autonomous > really_bad_threshold) {
      ++num_autonomous_really_bad_forward;
    }
    if (dist_autonomous > really_really_bad_threshold) {
      ++num_autonomous_really_really_bad_forward;
    }
    vec3 agranovsky_sample;
    try {
      agranovsky_sample = agranovsky.evaluate_full_forward(p.x1());
    } catch (std::exception&) {
      ++num_autonomous_better_forward;
      continue;
    }
    auto const dist_agranovsky = distance(numerical_x0, agranovsky_sample);
    if (dist_autonomous < dist_agranovsky) {
      ++num_autonomous_better_forward;
    }
    if (dist_agranovsky > bad_threshold) {
      ++num_agranovsky_bad_forward;
    }
    if (dist_agranovsky > really_bad_threshold) {
      ++num_agranovsky_really_bad_forward;
    }
    if (dist_agranovsky > really_really_bad_threshold) {
      ++num_agranovsky_really_really_bad_forward;
    }
  }
  report << std::defaultfloat
         << (double)num_autonomous_better_forward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in forward are better\n"
         << (double)num_autonomous_bad_forward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in forward are bad\n"
         << (double)num_autonomous_really_bad_forward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in forward are really bad\n"
         << (double)num_autonomous_really_really_bad_forward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in forward are really, really bad\n"
         << (double)num_agranovsky_bad_forward /
                (double)advected_particles.size() * 100
         << "% agranovsky in forward is bad\n"
         << (double)num_agranovsky_really_bad_forward /
                (double)advected_particles.size() * 100
         << "% agranovsky in forward is really bad\n"
         << (double)num_agranovsky_really_really_bad_forward /
                (double)advected_particles.size() * 100
         << "% agranovsky in forward is really, really bad\n\n";

      return std::tuple{num_autonomous_better_forward,
                        num_autonomous_bad_forward,
                        num_autonomous_really_bad_forward,
                        num_autonomous_really_really_bad_forward,
                        num_agranovsky_bad_forward,
                        num_agranovsky_really_bad_forward,
                        num_agranovsky_really_really_bad_forward};
}
//------------------------------------------------------------------------------
auto compare_direct_backward(auto const& advected_particles,
                             auto&& numerical_flowmap, auto&& agranovsky,
                             auto const& args, auto& report) {
  size_t num_autonomous_better_backward            = 0;
  size_t num_autonomous_bad_backward               = 0;
  size_t num_autonomous_really_bad_backward        = 0;
  size_t num_autonomous_really_really_bad_backward = 0;
  size_t num_agranovsky_bad_backward               = 0;
  size_t num_agranovsky_really_bad_backward        = 0;
  size_t num_agranovsky_really_really_bad_backward = 0;
  for (auto const& p : advected_particles) {
    auto const numerical_x0 =
        numerical_flowmap(p.x1(), args.t0 + args.tau, -args.tau);
    auto const dist_autonomous = distance(numerical_x0, p.x0());

    if (dist_autonomous > bad_threshold) {
      ++num_autonomous_bad_backward;
    }
    if (dist_autonomous > really_bad_threshold) {
      ++num_autonomous_really_bad_backward;
    }
    if (dist_autonomous > really_really_bad_threshold) {
      ++num_autonomous_really_really_bad_backward;
    }
    vec3 agranovsky_sample;
    try {
      agranovsky_sample = agranovsky.evaluate_full_backward(p.x1());
    } catch (std::exception&) {
      ++num_autonomous_better_backward;
      continue;
    }
    auto const dist_agranovsky = distance(numerical_x0, agranovsky_sample);
    if (dist_autonomous < dist_agranovsky) {
      ++num_autonomous_better_backward;
    }

    if (dist_agranovsky > bad_threshold) {
      ++num_agranovsky_bad_backward;
    }
    if (dist_agranovsky > really_bad_threshold) {
      ++num_agranovsky_really_bad_backward;
    }
    if (dist_agranovsky > really_really_bad_threshold) {
      ++num_agranovsky_really_really_bad_backward;
    }
  }
  report << std::defaultfloat
         << (double)num_autonomous_better_backward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in backward are better\n"
         << (double)num_autonomous_bad_backward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in backward are bad\n"
         << (double)num_autonomous_really_bad_backward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in backward are really bad\n"
         << (double)num_autonomous_really_really_bad_backward /
                (double)advected_particles.size() * 100
         << "% autonomous particles in backward are really, really bad\n"
         << (double)num_agranovsky_bad_backward /
                (double)advected_particles.size() * 100
         << "% agranovsky in backward is bad\n"
         << (double)num_agranovsky_really_bad_backward /
                (double)advected_particles.size() * 100
         << "% agranovsky in backward is really bad\n"
         << (double)num_agranovsky_really_really_bad_backward /
                (double)advected_particles.size() * 100
         << "% agranovsky in backward is really, really bad\n\n";

      return std::tuple{num_autonomous_better_backward,
                        num_autonomous_bad_backward,
                        num_autonomous_really_bad_backward,
                        num_autonomous_really_really_bad_backward,
                        num_agranovsky_bad_backward,
                        num_agranovsky_really_bad_backward,
                        num_agranovsky_really_really_bad_backward};
}

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

  analytical::fields::numerical::tornado v;

  auto calc_particles = [&args, &v](auto const& initial_particles) ->
      typename std::decay_t<decltype(initial_particles.front())>::container_t {
        switch (args.num_splits) {
          // case 2:
          //  return initial_particles.front().advect_with_2_splits(
          //      args.tau_step, args.t0 + args.tau, args.max_num_particles,
          //      initial_particles);
          case 3:
            return initial_particles.front().advect_with_3_splits(
                flowmap(v), args.tau_step, args.t0 + args.tau,
                args.max_num_particles, initial_particles);
            // case 5:
            //  return initial_particles.front().advect_with_5_splits(
            //      args.tau_step, args.t0 + args.tau, args.max_num_particles,
            //      initial_particles);
            // case 7:
            //  return initial_particles.front().advect_with_7_splits(
            //      args.tau_step, args.t0 + args.tau, args.max_num_particles,
            //      initial_particles);
        }
        return {};
      };


  indeterminate_progress_bar([&](auto indicator) {
    indicator.set_text("Building numerical flowmap");
    auto numerical_flowmap = flowmap(v);
    numerical_flowmap.use_caching(false);

    indicator.set_text("Building initial particle list");
    auto initial_particles = create_initial_particles(args);
    report << "number of initial particles: " << initial_particles.size()
           << '\n';

    indicator.set_text("Advecting autonomous particles");
    auto const advected_particles = calc_particles(initial_particles);
    report << "number of advected particles: " << advected_particles.size()
           << '\n';
    
    indicator.set_text("Writing ellipses to NetCDF");
    if (args.write_ellipses_to_netcdf) {
      write_x1(initial_particles, "tornado_grid_initial.nc");
      write_x1(advected_particles, "tornado_grid_advected.nc");
      write_x0(advected_particles, "tornado_grid_back_calculation.nc");
    }

    indicator.set_text("Inserting particles into mesh");
    auto  autonomous_mesh = create_autonomous_mesh(advected_particles);
    auto& autonomous_flowmap_mesh_prop =
        autonomous_mesh.vertex_property<vec3>("flowmap");

    indicator.set_text("Building delaunay mesh");
    autonomous_mesh.build_delaunay_mesh();

    indicator.set_text("Writing delaunay meshed flowmap");
    autonomous_mesh.write_vtk("tornado_autonomous_forward_flowmap.vtk");

    indicator.set_text("Creating Sampler");
    // auto flowmap_sampler_autonomous_particles =
    //    autonomous_mesh.sampler(autonomous_flowmap_mesh_prop);
    [[maybe_unused]] auto flowmap_sampler_autonomous_particles =
        autonomous_mesh.inverse_distance_weighting_sampler(
            autonomous_flowmap_mesh_prop);
    auto const grid_min_extent =
        static_cast<size_t>(std::ceil(std::sqrt(size(advected_particles) / 2)));
    //----------------------------------------------------------------------------
    // build agranovsky
    //----------------------------------------------------------------------------
    indicator.set_text("Building agranovsky flowmap");
    auto agranovsky = create_agranovsky_flowmap(v, grid_min_extent, args);
    //----------------------------------------------------------------------------
    // comparing back calculations with agranovsky
    //----------------------------------------------------------------------------
    indicator.set_text("Comparing direct positions forward");
    [[maybe_unused]] auto const [num_autonomous_better_forward,
                                 num_autonomous_bad_forward,
                                 num_autonomous_really_bad_forward,
                                 num_autonomous_really_really_bad_forward,
                                 num_agranovsky_bad_forward,
                                 num_agranovsky_really_bad_forward,
                                 num_agranovsky_really_really_bad_forward] =
        compare_direct_forward(advected_particles, numerical_flowmap,
                               agranovsky, args, report);

    indicator.set_text("Comparing direct positions backward");
    [[maybe_unused]] auto const [num_autonomous_better_backward,
                                 num_autonomous_bad_backward,
                                 num_autonomous_really_bad_backward,
                                 num_autonomous_really_really_bad_backward,
                                 num_agranovsky_bad_backward,
                                 num_agranovsky_really_bad_backward,
                                 num_agranovsky_really_really_bad_backward] =
        compare_direct_backward(advected_particles, numerical_flowmap,
                                agranovsky, args, report);
    //----------------------------------------------------------------------------
    // build flowmap on uniform grid that has approximately the same number of
    // vertices as particles
    //----------------------------------------------------------------------------
    indicator.set_text("Buildung regular sampled flowmap");
    rectilinear_grid uniform_rectilinear_grid{
        linspace{0.0, 2.0, grid_min_extent * 2},
        linspace{0.0, 1.0, grid_min_extent}};
    auto& regular_flowmap_grid_prop =
        uniform_rectilinear_grid.add_vertex_property<vec3, x_fastest>(
            "flowmap");
    uniform_rectilinear_grid.iterate_over_vertex_indices([&](auto const... is) {
      regular_flowmap_grid_prop(is...) =
          numerical_flowmap(uniform_rectilinear_grid(is...), args.t0, args.tau);
    });
    unstructured_tetrahedral_grid<double, 3> regular_mesh{
        uniform_rectilinear_grid};
    auto&                      regular_flowmap_mesh_prop =
        regular_mesh.vertex_property<vec3>("flowmap");
    regular_mesh.write_vtk("tornado_regular_forward_flowmap.vtk");

    // auto flowmap_sampler_regular =
    //    regular_mesh.sampler(regular_flowmap_mesh_prop);
    [[maybe_unused]] auto flowmap_sampler_regular =
        regular_mesh.inverse_distance_weighting_sampler(
            regular_flowmap_mesh_prop);

    //----------------------------------------------------------------------------
    // Create memory for measuring
    //----------------------------------------------------------------------------
    rectilinear_grid  sampler_check_grid{linspace{0.0, 2.0, 101}, linspace{0.0, 1.0, 51}};
    [[maybe_unused]] auto& forward_errors_autonomous_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_autonomous");
    [[maybe_unused]] auto& forward_errors_regular_prop =
        sampler_check_grid.add_vertex_property<double>("forward_error_regular");
    [[maybe_unused]] auto& forward_errors_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_agranovsky");
    [[maybe_unused]] auto& forward_errors_diff_regular_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_diff_regular");
    [[maybe_unused]] auto& forward_errors_diff_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_diff_agranovsky");

    [[maybe_unused]] auto& backward_errors_autonomous_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_autonomous");
    [[maybe_unused]] auto& backward_errors_regular_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_regular");
    [[maybe_unused]] auto& backward_errors_diff_regular_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_diff_regular");
    [[maybe_unused]] auto& backward_errors_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_agranovsky");
    [[maybe_unused]] auto& backward_errors_diff_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_diff_agranovsky");

    double mean_autonomous_forward_error  = std::numeric_limits<double>::max(),
           mean_regular_forward_error     = std::numeric_limits<double>::max(),
           mean_agranovsky_forward_error  = std::numeric_limits<double>::max();
    double mean_autonomous_backward_error = std::numeric_limits<double>::max(),
           mean_regular_backward_error    = std::numeric_limits<double>::max(),
           mean_agranovsky_backward_error = std::numeric_limits<double>::max();
    size_t num_points_ood_forward = 0, num_points_ood_backward = 0;

    ////----------------------------------------------------------------------------
    //// Compare forward
    ////----------------------------------------------------------------------------
    //indicator.set_text("Comparing forward advection");
    //{
    //  std::vector<double> autonomous_errors, regular_errors, agranovsky_errors;
    //  autonomous_errors.reserve(sampler_check_grid.vertices().size());
    //  regular_errors.reserve(sampler_check_grid.vertices().size());
    //  agranovsky_errors.reserve(sampler_check_grid.vertices().size());
    //  sampler_check_grid.iterate_over_vertex_indices([&](auto const... is) {
    //    auto x = sampler_check_grid(is...);
    //    try {
    //      auto const autonomous_advection =
    //          flowmap_sampler_autonomous_particles(x);
    //      auto const regular_advection    = flowmap_sampler_regular(x);
    //      auto const agranovsky_advection = agranovsky.evaluate_full_forward(x);
    //      auto const numerical_advection =
    //          numerical_flowmap(x, args.t0, args.tau);
    //      autonomous_errors.push_back(
    //          distance(autonomous_advection, numerical_advection));
    //      forward_errors_autonomous_prop(is...) = autonomous_errors.back();
    //
    //      regular_errors.push_back(
    //          distance(regular_advection, numerical_advection));
    //      forward_errors_regular_prop(is...) = regular_errors.back();
    //
    //      agranovsky_errors.push_back(
    //          distance(agranovsky_advection, numerical_advection));
    //      forward_errors_agranovsky_prop(is...) = agranovsky_errors.back();
    //
    //      forward_errors_diff_regular_prop(is...) =
    //          forward_errors_regular_prop(is...) -
    //          forward_errors_autonomous_prop(is...);
    //      forward_errors_diff_agranovsky_prop(is...) =
    //          forward_errors_agranovsky_prop(is...) -
    //          forward_errors_autonomous_prop(is...);
    //    } catch (std::exception const& e) {
    //      ++num_points_ood_forward;
    //      forward_errors_autonomous_prop(is...)      = 0.0 / 0.0;
    //      forward_errors_regular_prop(is...)         = 0.0 / 0.0;
    //      forward_errors_agranovsky_prop(is...)      = 0.0 / 0.0;
    //      forward_errors_diff_regular_prop(is...)    = 0.0 / 0.0;
    //      forward_errors_diff_agranovsky_prop(is...) = 0.0 / 0.0;
    //    }
    //  });
    //  mean_autonomous_forward_error =
    //      std::accumulate(begin(autonomous_errors), end(autonomous_errors),
    //                      double(0)) /
    //      size(autonomous_errors);
    //  mean_regular_forward_error =
    //      std::accumulate(begin(regular_errors), end(regular_errors),
    //                      double(0)) /
    //      size(regular_errors);
    //  mean_agranovsky_forward_error =
    //      std::accumulate(begin(agranovsky_errors), end(agranovsky_errors),
    //                      double(0)) /
    //      size(agranovsky_errors);
    //}
    //
    //----------------------------------------------------------------------------
    // Reverse Meshes
    //----------------------------------------------------------------------------
    indicator.set_text("Reversing Meshes");
     for (auto v : autonomous_mesh.vertices()) {
      std::swap(autonomous_mesh[v](0), autonomous_flowmap_mesh_prop[v](0));
      std::swap(autonomous_mesh[v](1), autonomous_flowmap_mesh_prop[v](1));
    }
     autonomous_mesh.build_delaunay_mesh();
     autonomous_mesh.write_vtk("tornado_autonomous_backward_flowmap.vtk");
     autonomous_mesh.rebuild_kd_tree();
    //autonomous_mesh.build_hierarchy();

     for (auto v : regular_mesh.vertices()) {
      std::swap(regular_mesh[v](0), regular_flowmap_mesh_prop[v](0));
      std::swap(regular_mesh[v](1), regular_flowmap_mesh_prop[v](1));
    }
     regular_mesh.build_delaunay_mesh();
     regular_mesh.build_hierarchy();
     regular_mesh.rebuild_kd_tree();
     regular_mesh.write_vtk("tornado_regular_backward_flowmap.vtk");

    //////----------------------------------------------------------------------------
    ////// Compare with backward advection
    //////----------------------------------------------------------------------------
    // indicator.set_text("Comparing backward advection");
    //{
    //  std::vector<double> autonomous_errors, regular_errors,
    //  agranovsky_errors;
    //  autonomous_errors.reserve(sampler_check_grid.vertices().size());
    //  regular_errors.reserve(sampler_check_grid.vertices().size());
    //  agranovsky_errors.reserve(sampler_check_grid.vertices().size());
    //  sampler_check_grid.iterate_over_vertex_indices([&](auto const... is) {
    //    auto x = sampler_check_grid(is...);
    //    try {
    //      // numerical backward advection
    //      auto const numerical_advection =
    //          numerical_flowmap(x, args.t0 + args.tau, -args.tau);
    //
    //      // autonomous backward advection
    //      auto const autonomous_advection =
    //          flowmap_sampler_autonomous_particles(x);
    //      autonomous_errors.push_back(
    //          distance(autonomous_advection, numerical_advection));
    //      backward_errors_autonomous_prop(is...) = autonomous_errors.back();
    //
    //      // regular backward advection
    //      auto const regular_advection = flowmap_sampler_regular(x);
    //      regular_errors.push_back(
    //          distance(regular_advection, numerical_advection));
    //      backward_errors_regular_prop(is...) = regular_errors.back();
    //      backward_errors_diff_regular_prop(is...) =
    //          backward_errors_regular_prop(is...) -
    //          backward_errors_autonomous_prop(is...);
    //
    //      auto const agranovsky_advection =
    //      agranovsky.evaluate_full_backward(x); agranovsky_errors.push_back(
    //          distance(agranovsky_advection, numerical_advection));
    //      backward_errors_agranovsky_prop(is...) = agranovsky_errors.back();
    //
    //      backward_errors_diff_agranovsky_prop(is...) =
    //          backward_errors_agranovsky_prop(is...) -
    //          backward_errors_autonomous_prop(is...);
    //    } catch (std::exception const& e) {
    //      ++num_points_ood_backward;
    //      backward_errors_autonomous_prop(is...)      = 0.0 / 0.0;
    //      backward_errors_regular_prop(is...)         = 0.0 / 0.0;
    //      backward_errors_agranovsky_prop(is...)      = 0.0 / 0.0;
    //      backward_errors_diff_regular_prop(is...)    = 0.0 / 0.0;
    //      backward_errors_diff_agranovsky_prop(is...) = 0.0 / 0.0;
    //    }
    //  });
    //  mean_regular_backward_error =
    //      std::accumulate(begin(regular_errors), end(regular_errors),
    //                      double(0)) /
    //      size(regular_errors);
    //  mean_autonomous_backward_error =
    //      std::accumulate(begin(autonomous_errors), end(autonomous_errors),
    //                      double(0)) /
    //      size(autonomous_errors);
    //  mean_agranovsky_backward_error =
    //      std::accumulate(begin(agranovsky_errors), end(agranovsky_errors),
    //                      double(0)) /
    //      size(agranovsky_errors);
    //}



    indicator.mark_as_completed();
     report            << num_points_ood_forward << " / " <<
        sampler_check_grid.vertices().size()
        << " out of domain in forward direction("
        << (100 * num_points_ood_forward /
            (double)sampler_check_grid.vertices().size())
        << "%)\n"
        << num_points_ood_backward << " / " <<
        sampler_check_grid.vertices().size()
        << " out of domain in backward direction("
        << (100 * num_points_ood_backward /
            (double)sampler_check_grid.vertices().size())
        << "%)\n"
        << "mean error forward autonomous particles: " << std::scientific
        << mean_autonomous_forward_error << '\n'
        << "mean error forward regular grid: " << std::scientific
        << mean_regular_forward_error << '\n'
        << "mean error forward agranovsky grid: " << std::scientific
        << mean_agranovsky_forward_error << '\n'
        << "mean error backward autonomous particles: " << std::scientific
        << mean_autonomous_backward_error << '\n'
        << "mean error backward regular grid: " << std::scientific
        << mean_regular_backward_error << '\n'
        << "mean error backward agranovsky grid: " << std::scientific
        << mean_agranovsky_backward_error << '\n';
     if (mean_regular_forward_error > mean_autonomous_forward_error &&
        mean_agranovsky_forward_error > mean_autonomous_forward_error) {
      report << "autonomous particles are better in forward direction\n";
    } else if (mean_agranovsky_forward_error > mean_regular_forward_error
    &&
               mean_autonomous_forward_error > mean_regular_forward_error)
               {
      report << "regular grid is better in forward direction\n";
    } else if (mean_regular_forward_error > mean_agranovsky_forward_error
    &&
               mean_autonomous_forward_error >
               mean_agranovsky_forward_error) {
      report << "agranovsky is better in forward direction\n";
    }
     if (mean_regular_backward_error > mean_autonomous_backward_error &&
        mean_agranovsky_backward_error > mean_autonomous_backward_error) {
      report << "autonomous particles are better in backward direction\n";
    } else if (mean_agranovsky_backward_error >
     mean_regular_backward_error &&
               mean_autonomous_backward_error >
               mean_regular_backward_error) {
      report << "regular grid is better in backward direction\n";
    } else if (mean_regular_backward_error >
     mean_agranovsky_backward_error &&
               mean_autonomous_backward_error >
                   mean_agranovsky_backward_error) {
      report << "agranovsky is better in backward direction\n";
    }
     sampler_check_grid.write("tornado_grid_errors.vtk");

  });
  std::cerr << report.str();
}
