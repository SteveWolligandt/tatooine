#include <tatooine/analytical/fields/numerical/doublegyre.h>
#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/grid.h>
#include <tatooine/netcdf.h>
#include <tatooine/progress_bars.h>
#include <tatooine/triangular_mesh.h>
#include <tatooine/vtk_legacy.h>
#include <tatooine/flowmap_agranovsky.h>

#include <iomanip>

#include "parse_args.h"
//==============================================================================
std::vector<size_t> const cnt{1, 2, 3};
std::vector<std::thread>  writers;
std::mutex                writer_mutex;
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto args_opt = parse_args(argc, argv);
  if (!args_opt) {
    return 1;
  }
  auto args           = *args_opt;
  auto calc_particles = [&args](auto const& initial_particles)
      -> std::vector<std::decay_t<decltype(initial_particles.front())>> {
    switch (args.num_splits) {
      case 2:
        return initial_particles.front().advect_with_2_splits(
            args.tau_step, args.t0 + args.tau, args.max_num_particles, initial_particles);
      case 3:
        return initial_particles.front().advect_with_3_splits(
            args.tau_step, args.t0 + args.tau, args.max_num_particles, initial_particles);
      case 5:
        return initial_particles.front().advect_with_5_splits(
            args.tau_step, args.t0 + args.tau, args.max_num_particles, initial_particles);
      case 7:
        return initial_particles.front().advect_with_7_splits(
            args.tau_step, args.t0 + args.tau, args.max_num_particles, initial_particles);
    }
    return {};
  };

  std::vector<size_t> initial_netcdf_is{0, 0, 0}, advected_netcdf_is{0, 0, 0},
      back_calculation_netcdf_is{0, 0, 0};
  netcdf::file initial_ellipsis_file{"doublegyre_grid_initial.nc",
                                     netCDF::NcFile::replace},
      advected_file{"doublegyre_grid_advected.nc", netCDF::NcFile::replace},
      back_calculation_file{"doublegyre_grid_back_calculation.nc",
                            netCDF::NcFile::replace};
  auto initial_var = initial_ellipsis_file.add_variable<float>(
      "transformations", {initial_ellipsis_file.add_dimension("index"),
                          initial_ellipsis_file.add_dimension("row", 2),
                          initial_ellipsis_file.add_dimension("column", 3)});

  auto advected_var = advected_file.add_variable<float>(
      "transformations", {advected_file.add_dimension("index"),
                          advected_file.add_dimension("row", 2),
                          advected_file.add_dimension("column", 3)});
  auto back_calculation_var = back_calculation_file.add_variable<float>(
      "transformations", {back_calculation_file.add_dimension("index"),
                          back_calculation_file.add_dimension("row", 2),
                          back_calculation_file.add_dimension("column", 3)});

  grid initial_autonomous_particles_grid{linspace{0, 2.0, args.width + 1},
                                         linspace{0, 1.0, args.height + 1}};
  initial_autonomous_particles_grid.dimension<0>().pop_front();
  initial_autonomous_particles_grid.dimension<1>().pop_front();
  auto const spacing_x =
      initial_autonomous_particles_grid.dimension<0>().spacing();
  auto const spacing_y =
      initial_autonomous_particles_grid.dimension<1>().spacing();
  initial_autonomous_particles_grid.dimension<0>().front() -= spacing_x / 2;
  initial_autonomous_particles_grid.dimension<0>().back() -= spacing_x / 2;
  initial_autonomous_particles_grid.dimension<1>().front() -= spacing_y / 2;
  initial_autonomous_particles_grid.dimension<1>().back() -= spacing_y / 2;
  double const r0 =
      initial_autonomous_particles_grid.dimension<0>().spacing() / 2;
  triangular_mesh<double, 2> autonomous_mesh;
  auto&                      autonomous_flowmap_mesh_prop =
      autonomous_mesh.add_vertex_property<vec<double, 2>>("flowmap");

  analytical::fields::numerical::doublegyre v;
  v.set_infinite_domain(true);

  indeterminate_progress_bar([&](auto indicator) {
    //----------------------------------------------------------------------------
    // build initial particle list
    //----------------------------------------------------------------------------
    indicator.set_text("Building initial particle list");
    std::vector<autonomous_particle<decltype(flowmap(v))>> initial_particles;
    for (auto const& x : initial_autonomous_particles_grid.vertices()) {
      initial_particles.emplace_back(v, x, args.t0, r0)
          .phi()
          .use_caching(false);
    }
    initial_autonomous_particles_grid.dimension<0>().front() += spacing_x / 2;
    initial_autonomous_particles_grid.dimension<0>().back() += spacing_x / 2;
    initial_autonomous_particles_grid.dimension<1>().front() += spacing_y / 2;
    initial_autonomous_particles_grid.dimension<1>().back() += spacing_y / 2;
    initial_autonomous_particles_grid.dimension<0>().pop_back();
    initial_autonomous_particles_grid.dimension<1>().pop_back();
    for (auto const& x : initial_autonomous_particles_grid.vertices()) {
      initial_particles.emplace_back(v, x, args.t0, r0)
          .phi()
          .use_caching(false);
    }

    //----------------------------------------------------------------------------
    // integrate particles
    //----------------------------------------------------------------------------
    indicator.set_text("Integrating autonomous particles");
    auto const advected_particles = calc_particles(initial_particles);
    autonomous_mesh.vertex_data().reserve(autonomous_mesh.num_vertices() +
                                          size(advected_particles));
    indicator.set_text("Writing discretized flowmap");
    for (auto const& p : advected_particles) {
      auto v                          = autonomous_mesh.insert_vertex(p.x0());
      autonomous_flowmap_mesh_prop[v] = p.x1();
    }

    //----------------------------------------------------------------------------
    // write ellipses to netcdf
    //----------------------------------------------------------------------------
    indicator.set_text("Writing ellipses to NetCDF Files");
    for (auto const& ap : initial_particles) {
      mat23f T{{ap.S()(0, 0), ap.S()(0, 1), ap.x1()(0)},
               {ap.S()(1, 0), ap.S()(1, 1), ap.x1()(1)}};
      initial_var.write(initial_netcdf_is, cnt, T.data_ptr());
      ++initial_netcdf_is.front();
    }
    for (auto const& p : advected_particles) {
      auto sqrS =
          inv(p.nabla_phi1()) * p.S() * p.S() * inv(transposed(p.nabla_phi1()));
      auto [eig_vecs, eig_vals] = eigenvectors_sym(sqrS);
      eig_vals = {std::sqrt(eig_vals(0)), std::sqrt(eig_vals(1))};
      transposed(eig_vecs);
      if (args.min_cond > 0 && eig_vals(1) / eig_vals(0) < args.min_cond) {
        continue;
      }

      // advection
      mat23f advected_T{{p.S()(0, 0), p.S()(0, 1), p.x1()(0)},
                        {p.S()(1, 0), p.S()(1, 1), p.x1()(1)}};
      {
        std::lock_guard lock{writer_mutex};
        advected_var.write(advected_netcdf_is, cnt, advected_T.data_ptr());
        ++advected_netcdf_is.front();
      }

      // back calculation
      auto   Sback = eig_vecs * diag(eig_vals) * transposed(eig_vecs);
      mat23f back_calculation_T{{Sback(0, 0), Sback(0, 1), p.x0()(0)},
                                {Sback(1, 0), Sback(1, 1), p.x0()(1)}};
      {
        std::lock_guard lock{writer_mutex};
        back_calculation_var.write(back_calculation_netcdf_is, cnt,
                                   back_calculation_T.data_ptr());
        ++back_calculation_netcdf_is.front();
      }
    }

    //----------------------------------------------------------------------------
    // build numerical flowmap
    //----------------------------------------------------------------------------
    indicator.set_text("Building numerical flowmap");
    auto numerical_flowmap = flowmap(v);
    numerical_flowmap.use_caching(false);

    //----------------------------------------------------------------------------
    // build samplable discretized flowmap
    //----------------------------------------------------------------------------
    indicator.set_text("Building delaunay triangulation");
    autonomous_mesh.triangulate_delaunay();
    indicator.set_text("Writing delaunay triangulated flowmap");
    autonomous_mesh.write_vtk("doublegyre_autonomous_forward_flowmap.vtk");
    indicator.set_text("Creating Sampler");
    auto flowmap_sampler_autonomous_particles =
        autonomous_mesh.sampler(autonomous_flowmap_mesh_prop);

    //----------------------------------------------------------------------------
    // build flowmap on uniform grid that has approximately the same number of
    // vertices as particles
    //----------------------------------------------------------------------------
    indicator.set_text("Buildung regular sampled flowmap");
    auto const n =
        static_cast<size_t>(std::ceil(std::sqrt(size(advected_particles) / 2)));
    grid  uniform_grid{linspace{0.0, 2.0, n * 2}, linspace{0.0, 1.0, n}};
    auto& regular_flowmap_grid_prop =
        uniform_grid.add_vertex_property<vec2, x_fastest>("flowmap");
    uniform_grid.loop_over_vertex_indices([&](auto const... is) {
      regular_flowmap_grid_prop(is...) =
          numerical_flowmap(uniform_grid(is...), args.t0, args.tau);
    });
    triangular_mesh<double, 2> regular_mesh{uniform_grid};
    auto&                      regular_flowmap_mesh_prop =
        regular_mesh.vertex_property<vec<double, 2>>("flowmap");
    regular_mesh.write_vtk("doublegyre_regular_forward_flowmap.vtk");

    auto regular_flowmap_sampler =
        regular_mesh.sampler(regular_flowmap_mesh_prop);

    //----------------------------------------------------------------------------
    // build agranovsky
    //----------------------------------------------------------------------------
    indicator.set_text("Building agranovsky flowmap");
    double const       agranovsky_delta_t = 1;
    flowmap_agranovsky agranovsky{
        v,          args.t0,    args.tau, agranovsky_delta_t,
        vec2{0, 0}, vec2{2, 1}, n * 2,    n};

    //----------------------------------------------------------------------------
    // Create memory for measuring
    //----------------------------------------------------------------------------
    grid  sampler_check_grid{linspace{0.0, 2.0, 1001}, linspace{0.0, 1.0, 501}};
    auto& forward_errors_autonomous_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_autonomous");
    auto& forward_errors_regular_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_regular");
    auto& forward_errors_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_agranovsky");
    auto& forward_errors_diff_regular_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_diff_regular");
    auto& forward_errors_diff_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "forward_error_diff_agranovsky");

    auto& backward_errors_autonomous_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_autonomous");
    auto& backward_errors_regular_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_regular");
    auto& backward_errors_diff_regular_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_diff_regular");
    auto& backward_errors_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_agranovsky");
    auto& backward_errors_diff_agranovsky_prop =
        sampler_check_grid.add_vertex_property<double>(
            "backward_error_diff_agranovsky");

    double mean_autonomous_forward_error  = std::numeric_limits<double>::max(),
           mean_regular_forward_error     = std::numeric_limits<double>::max(),
           mean_agranovsky_forward_error  = std::numeric_limits<double>::max();
    double mean_autonomous_backward_error = std::numeric_limits<double>::max(),
           mean_regular_backward_error    = std::numeric_limits<double>::max(),
           mean_agranovsky_backward_error = std::numeric_limits<double>::max();
    size_t num_points_ood_forward = 0, num_points_ood_backward = 0;

    //----------------------------------------------------------------------------
    // Compare forward
    //----------------------------------------------------------------------------
    indicator.set_text("Comparing forward advection");
    {
      std::vector<double> autonomous_errors, regular_errors, agranovsky_errors;
      autonomous_errors.reserve(sampler_check_grid.num_vertices());
      regular_errors.reserve(sampler_check_grid.num_vertices());
      agranovsky_errors.reserve(sampler_check_grid.num_vertices());
      sampler_check_grid.loop_over_vertex_indices([&](auto const... is) {
        auto x = sampler_check_grid(is...);
        try {
          auto const autonomous_advection =
              flowmap_sampler_autonomous_particles(x);
          auto const regular_advection = regular_flowmap_sampler(x);
          auto const agranovsky_advection = agranovsky.evaluate_full_forward(x);
          auto const numerical_advection =
              numerical_flowmap(x, args.t0, args.tau);
          autonomous_errors.push_back(
              distance(autonomous_advection, numerical_advection));
          forward_errors_autonomous_prop(is...) = autonomous_errors.back();

          regular_errors.push_back(
              distance(regular_advection, numerical_advection));
          forward_errors_regular_prop(is...) = regular_errors.back();

          agranovsky_errors.push_back(
              distance(agranovsky_advection, numerical_advection));
          forward_errors_agranovsky_prop(is...) = agranovsky_errors.back();

          forward_errors_diff_regular_prop(is...) =
              forward_errors_regular_prop(is...) -
              forward_errors_autonomous_prop(is...);
          forward_errors_diff_agranovsky_prop(is...) =
              forward_errors_agranovsky_prop(is...) -
              forward_errors_autonomous_prop(is...);
        } catch (std::exception const& e) {
          ++num_points_ood_forward;
          forward_errors_autonomous_prop(is...)      = 0.0 / 0.0;
          forward_errors_regular_prop(is...)         = 0.0 / 0.0;
          forward_errors_agranovsky_prop(is...)      = 0.0 / 0.0;
          forward_errors_diff_regular_prop(is...)    = 0.0 / 0.0;
          forward_errors_diff_agranovsky_prop(is...) = 0.0 / 0.0;
        }
      });
      mean_autonomous_forward_error =
          std::accumulate(begin(autonomous_errors), end(autonomous_errors),
                          double(0)) /
          size(autonomous_errors);
      mean_regular_forward_error =
          std::accumulate(begin(regular_errors), end(regular_errors),
                          double(0)) /
          size(regular_errors);
      mean_agranovsky_forward_error =
          std::accumulate(begin(agranovsky_errors), end(agranovsky_errors),
                          double(0)) /
          size(agranovsky_errors);
    }

    //----------------------------------------------------------------------------
    // Reverse Meshes
    //----------------------------------------------------------------------------
    for (auto v : autonomous_mesh.vertices()) {
      std::swap(autonomous_mesh[v](0), autonomous_flowmap_mesh_prop[v](0));
      std::swap(autonomous_mesh[v](1), autonomous_flowmap_mesh_prop[v](1));
    }
    autonomous_mesh.triangulate_delaunay();
    autonomous_mesh.build_hierarchy();
    autonomous_mesh.write_vtk("doublegyre_autonomous_backward_flowmap.vtk");

    for (auto v : regular_mesh.vertices()) {
      std::swap(regular_mesh[v](0), regular_flowmap_mesh_prop[v](0));
      std::swap(regular_mesh[v](1), regular_flowmap_mesh_prop[v](1));
    }
    regular_mesh.triangulate_delaunay();
    regular_mesh.build_hierarchy();
    regular_mesh.write_vtk("doublegyre_regular_backward_flowmap.vtk");

    //----------------------------------------------------------------------------
    // Compare with backward advection
    //----------------------------------------------------------------------------
    indicator.set_text("Comparing backward advection");
    {
      std::vector<double> autonomous_errors, regular_errors, agranovsky_errors;
      autonomous_errors.reserve(sampler_check_grid.num_vertices());
      regular_errors.reserve(sampler_check_grid.num_vertices());
      agranovsky_errors.reserve(sampler_check_grid.num_vertices());
      sampler_check_grid.loop_over_vertex_indices([&](auto const... is) {
        auto x = sampler_check_grid(is...);
        try {
          // numerical backward advection
          auto const numerical_advection =
              numerical_flowmap(x, args.t0 + args.tau, -args.tau);

          // autonomous backward advection
          auto const autonomous_advection =
              flowmap_sampler_autonomous_particles(x);
          autonomous_errors.push_back(
              distance(autonomous_advection, numerical_advection));
          backward_errors_autonomous_prop(is...) = autonomous_errors.back();

          // regular backward advection
          auto const regular_advection = regular_flowmap_sampler(x);
          regular_errors.push_back(
              distance(regular_advection, numerical_advection));
          backward_errors_regular_prop(is...) = regular_errors.back();
          backward_errors_diff_regular_prop(is...) =
              backward_errors_regular_prop(is...) -
              backward_errors_autonomous_prop(is...);

          auto const agranovsky_advection = agranovsky.evaluate_full_backward(x);
          agranovsky_errors.push_back(
              distance(agranovsky_advection, numerical_advection));
          backward_errors_agranovsky_prop(is...) = agranovsky_errors.back();

          backward_errors_diff_agranovsky_prop(is...) =
              backward_errors_agranovsky_prop(is...) -
              backward_errors_autonomous_prop(is...);
        } catch (std::exception const& e) {
          ++num_points_ood_backward;
          backward_errors_autonomous_prop(is...)      = 0.0 / 0.0;
          backward_errors_regular_prop(is...)         = 0.0 / 0.0;
          backward_errors_agranovsky_prop(is...)      = 0.0 / 0.0;
          backward_errors_diff_regular_prop(is...)    = 0.0 / 0.0;
          backward_errors_diff_agranovsky_prop(is...) = 0.0 / 0.0;
        }
      });
      mean_regular_backward_error =
          std::accumulate(begin(regular_errors), end(regular_errors),
                          double(0)) /
          size(regular_errors);
      mean_autonomous_backward_error =
          std::accumulate(begin(autonomous_errors), end(autonomous_errors),
                          double(0)) /
          size(autonomous_errors);
      mean_agranovsky_backward_error =
          std::accumulate(begin(agranovsky_errors), end(agranovsky_errors),
                          double(0)) /
          size(agranovsky_errors);
    }
    indicator.mark_as_completed();
    std::cerr << '\n';
    std::cerr
        << "==============================================================="
           "=================\n"
        << "REPORT\n"
        << "==============================================================="
           "=================\n"
        << "number of initial particles: " << initial_particles.size() << '\n'
        << "number of advected particles: " << advected_particles.size() << '\n'
        << "number of grid vertices: " << n * 2 * n << '\n'
        << num_points_ood_forward << " / " << sampler_check_grid.num_vertices()
        << " out of domain in forward direction("
        << (100 * num_points_ood_forward /
            (double)sampler_check_grid.num_vertices())
        << "%)\n"
        << num_points_ood_backward << " / " << sampler_check_grid.num_vertices()
        << " out of domain in backward direction("
        << (100 * num_points_ood_backward /
            (double)sampler_check_grid.num_vertices())
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
      std::cerr << "autonomous particles are better in forward direction\n";
    } else if (mean_agranovsky_forward_error > mean_regular_forward_error &&
               mean_autonomous_forward_error > mean_regular_forward_error) {
      std::cerr << "regular grid is better in forward direction\n";
    } else if (mean_regular_forward_error > mean_agranovsky_forward_error &&
               mean_autonomous_forward_error > mean_agranovsky_forward_error) {
      std::cerr << "agranovsky is better in forward direction\n";
    }
    if (mean_regular_backward_error > mean_autonomous_backward_error &&
        mean_agranovsky_backward_error > mean_autonomous_backward_error) {
      std::cerr << "autonomous particles are better in backward direction\n";
    } else if (mean_agranovsky_backward_error > mean_regular_backward_error &&
               mean_autonomous_backward_error > mean_regular_backward_error) {
      std::cerr << "regular grid is better in backward direction\n";
    } else if (mean_regular_backward_error > mean_agranovsky_backward_error &&
               mean_autonomous_backward_error > mean_agranovsky_backward_error) {
      std::cerr << "agranovsky is better in backward direction\n";
    }
    sampler_check_grid.write("doublegyre_grid_errors.vtk");
  });

  for (auto& writer : writers) {
    writer.join();
  }
}
