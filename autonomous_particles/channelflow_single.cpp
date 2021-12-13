#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>

#include <iomanip>
#include <sstream>

#include "parse_args.h"
//==============================================================================
using namespace tatooine;
//==============================================================================
auto main(int argc, char** argv) -> int {
  indeterminate_progress_bar([&](auto indicator) {
    indicator.set_text("Parsing Arguments");
    auto args_opt = parse_args<3>(argc, argv);
    if (!args_opt) {
      return;
    }
    auto args = *args_opt;

    indicator.set_text("Opening File");
    auto channelflow_154_file =
        hdf5::file{args.velocity_file == std::nullopt
                       ? "/home/vcuser/channel_flow/dino_res_154000.h5"
                       : *args.velocity_file};
    auto discrete_channelflow_domain = nonuniform_rectilinear_grid3{};

    indicator.set_text("Reading Axes");
    channelflow_154_file.dataset<double>("CartGrid/axis0")
        .read(discrete_channelflow_domain.dimension<0>());
    //discrete_channelflow_domain.dimension<0>().pop_back();
    channelflow_154_file.dataset<double>("CartGrid/axis1")
        .read(discrete_channelflow_domain.dimension<1>());
    discrete_channelflow_domain.push_back<1>();
    channelflow_154_file.dataset<double>("CartGrid/axis2")
        .read(discrete_channelflow_domain.dimension<2>());
    discrete_channelflow_domain.push_back<2>();
    discrete_channelflow_domain.write("channelflow_grid.vtk");

    if (args.show_dimensions) {
      std::cerr << discrete_channelflow_domain << '\n';
      return;
    }

    indicator.set_text("Allocating data for velocity");
    auto& discrete_velocity = *dynamic_cast<
        typed_vertex_property<nonuniform_rectilinear_grid3, vec3,
                              dynamic_multidim_array<vec3, x_fastest>>*>(
        &discrete_channelflow_domain.vec3_vertex_property("velocity"));

    indicator.set_text("Creating sampler");
    auto       w = discrete_velocity.linear_sampler();
    auto const v = make_infinite<1, 2>(w);

    indicator.set_text("Loading x-velocity");
    {
      auto dataset    = channelflow_154_file.dataset<double>("velocity/xvel");
      auto data_space = dataset.dataspace();
      data_space.select_hyperslab({0, 0, 0},
                                  {discrete_channelflow_domain.size<0>(),
                                   discrete_channelflow_domain.size<1>() - 1,
                                   discrete_channelflow_domain.size<2>() - 1});
      auto mem_space =
          hdf5::dataspace{discrete_channelflow_domain.size<0>() * 3,
                          discrete_channelflow_domain.size<1>() - 1,
                          discrete_channelflow_domain.size<2>() - 1};
      mem_space.select_hyperslab({0, 0, 0}, {3, 1, 1},
                                 {discrete_channelflow_domain.size<0>(),
                                  discrete_channelflow_domain.size<1>() - 1,
                                  discrete_channelflow_domain.size<2>() - 1});
      dataset.read(mem_space.id(), data_space.id(), H5P_DEFAULT,
                   discrete_velocity.data().front().data_ptr());
    }

    indicator.set_text("Loading y-velocity");
    {
      auto dataset    = channelflow_154_file.dataset<double>("velocity/yvel");
      auto data_space = dataset.dataspace();
      data_space.select_hyperslab({0, 0, 0},
                                  {discrete_channelflow_domain.size<0>(),
                                   discrete_channelflow_domain.size<1>() - 1,
                                   discrete_channelflow_domain.size<2>() - 1});
      auto mem_space = hdf5::dataspace{discrete_channelflow_domain.size(0) * 3,
                                       discrete_channelflow_domain.size(1) - 1,
                                       discrete_channelflow_domain.size(2) - 1};
      mem_space.select_hyperslab({1, 0, 0}, {3, 1, 1},
                                 {discrete_channelflow_domain.size(0),
                                  discrete_channelflow_domain.size(1) - 1,
                                  discrete_channelflow_domain.size(2) - 1});

      dataset.read(mem_space.id(), data_space.id(), H5P_DEFAULT,
                   discrete_velocity.data().front().data_ptr());
    }
    indicator.set_text("Loading z-velocity");
    {
      auto dataset    = channelflow_154_file.dataset<double>("velocity/zvel");
      auto data_space = dataset.dataspace();
      data_space.select_hyperslab({0, 0, 0},
                                  {discrete_channelflow_domain.size<0>(),
                                   discrete_channelflow_domain.size<1>() - 1,
                                   discrete_channelflow_domain.size<2>() - 1});
      auto mem_space = hdf5::dataspace{discrete_channelflow_domain.size(0) * 3,
                                       discrete_channelflow_domain.size(1) - 1,
                                       discrete_channelflow_domain.size(2) - 1};
      mem_space.select_hyperslab({2, 0, 0}, {3, 1, 1},
                                 {discrete_channelflow_domain.size(0),
                                  discrete_channelflow_domain.size(1) - 1,
                                  discrete_channelflow_domain.size(2) - 1});

      dataset.read(mem_space.id(), data_space.id(), H5P_DEFAULT,
                   discrete_velocity.data().front().data_ptr());
    }
    indicator.set_text("Creating slabs for infinite domain");
    repeat_for_infinite<1, 2>(discrete_velocity);

    //----------------------------------------------------------------------------
    // indicator.set_text("Building numerical flowmap");
    auto phi = flowmap(v);
    phi.use_caching(false);
    //----------------------------------------------------------------------------
    indicator.set_text("Advecting autonomous particles");
    auto const initial_part = autonomous_particle3{args.x0, args.t0, args.r0};
    auto const advected_particles = [&] {
      switch (args.split_behavior) {
          //  case split_behavior_t::two_splits:
          //    return initial_part
          //        .advect<autonomous_particle3::split_setups::two_splits>(
          //            phi, args.step_width, args.tau);
        default:
        case split_behavior_t::three_splits:
          return initial_part
              .advect<autonomous_particle3::split_setups::three_splits>(
                  phi, args.step_width, args.tau);
          // case split_behavior_t::five_splits:
          //   return initial_part
          //       .advect<autonomous_particle3::split_setups::five_splits>(
          //           phi, args.step_width, args.tau);
          // case split_behavior_t::seven_splits:
          //   return initial_part
          //       .advect<autonomous_particle3::split_setups::seven_splits>(
          //           phi, args.step_width, args.tau);
          // case split_behavior_t::centered_four:
          //   return initial_part
          //       .advect<autonomous_particle3::split_setups::centered_four>(
          //           phi, args.step_width, args.tau);
      }
    }();
    std::cerr << "number of advected particles: " << size(advected_particles)
              << '\n';
    //----------------------------------------------------------------------------
    indicator.set_text("Writing Autonomous Particles Results");
    auto all_advected_discretizations =
        std::vector<unstructured_triangular_grid3>{};
    auto all_initial_discretizations =
        std::vector<unstructured_triangular_grid3>{};
    for (auto const& p : advected_particles) {
      all_initial_discretizations.push_back(discretize(p.initial_ellipse(), 2));
      all_advected_discretizations.push_back(discretize(p, 2));
    }
    all_initial_discretizations.front().write_vtp(
        "channelflow_single_front.vtp");
    write_vtk(all_initial_discretizations,
              "channelflow_single_ellipsoids0.vtk");
    write_vtk(all_advected_discretizations,
              "channelflow_single_ellipsoids1.vtk");
  });
}
