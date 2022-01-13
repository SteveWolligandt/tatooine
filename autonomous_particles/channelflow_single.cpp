#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/progress_bars.h>
#include <tatooine/vtk_legacy.h>

#include <iomanip>
#include <sstream>

#include "parse_args.h"
//==============================================================================
namespace tatooine::autonomous_particles {
auto channel_flow_single(args_t);
}  // namespace tatooine::autonomous_particles
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto args_opt = parse_args<3>(argc, argv);
  if (!args_opt) {
    return;
  }
  auto args = *args_opt;
  tatooine::autonomous_particles::channel_flow_single(args);
}
//==============================================================================
namespace tatooine::autonomous_particles {
//==============================================================================
auto channel_flow_single(args_t) {
  indeterminate_progress_bar([&](auto indicator) {
    indicator.set_text("Parsing Arguments");

    indicator.set_text("Opening File");
    auto channelflow_154_file =
        hdf5::file{args.velocity_file == std::nullopt
                       ? "/home/vcuser/channel_flow/dino_res_154000.h5"
                       : *args.velocity_file};
    auto discrete_channelflow_domain = nonuniform_rectilinear_grid3{};

    indicator.set_text("Reading Axes");
    channelflow_154_file.dataset<double>("CartGrid/axis0")
        .read(discrete_channelflow_domain.dimension<0>());
    // discrete_channelflow_domain.dimension<0>().pop_back();
    channelflow_154_file.dataset<double>("CartGrid/axis1")
        .read(discrete_channelflow_domain.dimension<1>());
    discrete_channelflow_domain.push_back<1>();
    channelflow_154_file.dataset<double>("CartGrid/axis2")
        .read(discrete_channelflow_domain.dimension<2>());
    discrete_channelflow_domain.push_back<2>();
    discrete_channelflow_domain.write("channelflow_grid.vtk");

    if (args.show_dimensions) {
      std::cerr << discrete_channelflow_domain << '\n';
      auto min_cell_extent = std::numeric_limits<real_t>::infinity();
      for (std::size_t i = 0; i < discrete_channelflow_domain.size<0>() - 1;
           ++i) {
        min_cell_extent =
            gcem::min(min_cell_extent,
                      discrete_channelflow_domain.extent_of_dimension<0>(i));
      }
      for (std::size_t i = 0; i < discrete_channelflow_domain.size<1>() - 1;
           ++i) {
        min_cell_extent =
            gcem::min(min_cell_extent,
                      discrete_channelflow_domain.extent_of_dimension<1>(i));
      }
      for (std::size_t i = 0; i < discrete_channelflow_domain.size<2>() - 1;
           ++i) {
        min_cell_extent =
            gcem::min(min_cell_extent,
                      discrete_channelflow_domain.extent_of_dimension<2>(i));
      }
      std::cout << "min cell extent: " << min_cell_extent << '\n';
      return;
    }

    indicator.set_text("Allocating data for velocity");
    auto& discrete_velocity =
        *dynamic_cast<nonuniform_rectilinear_grid3::typed_vertex_property_t<
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

    //--------------------------------------------------------------------------
    // indicator.set_text("Building numerical flowmap");
    auto phi = flowmap(v);
    phi.use_caching(false);
    //--------------------------------------------------------------------------
    indicator.set_text("Advecting autonomous particles");
    auto const [advected_particles, advected_simple_particles] = advect(
        autonomous_particle3{args.x0, args.t0, args.r0}, args.split_behavior);
    std::cerr << "number of advected particles: " << size(advected_particles)
              << '\n';
    std::cerr << "number of advected simple particles: "
              << size(advected_simple_particles) << '\n';
    //--------------------------------------------------------------------------
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
    write_vtp(all_initial_discretizations,
              "channelflow_single_ellipsoids0.vtp");
    write_vtp(all_advected_discretizations,
              "channelflow_single_ellipsoids1.vtp");
    x0_to_pointset(advected_simple_particles)
        .write_vtp("channelflow_simple_particles0.vtp");
    x1_to_pointset(advected_simple_particles)
        .write_vtp("channelflow_simple_particles1.vtp");
  });
}
//==============================================================================
}  // namespace tatooine::autonomous_particles
//==============================================================================
