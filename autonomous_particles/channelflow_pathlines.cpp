#include <tatooine/autonomous_particle.h>
#include <tatooine/chrono.h>
#include <tatooine/trace_flow.h>
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
      auto min_cell_extent = std::numeric_limits<real_type>::infinity();
      for (std::size_t i = 0; i < discrete_channelflow_domain.size<0>() - 1;
           ++i) {
        min_cell_extent = gcem::min(
            min_cell_extent,
            discrete_channelflow_domain.extent_of_dimension<0>(i));
      }
      for (std::size_t i = 0; i < discrete_channelflow_domain.size<1>() - 1;
           ++i) {
        min_cell_extent = gcem::min(
            min_cell_extent,
            discrete_channelflow_domain.extent_of_dimension<1>(i));
      }
      for (std::size_t i = 0; i < discrete_channelflow_domain.size<2>() - 1;
           ++i) {
        min_cell_extent = gcem::min(
            min_cell_extent,
            discrete_channelflow_domain.extent_of_dimension<2>(i));
      }
      std::cout << "min cell extent: " << min_cell_extent << '\n';



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
    std::cout << "num_repeated_dimensions: " << v.num_repeated_dimensions << '\n';
    std::cout << "repeated_dimensions: [" << v.repeated_dimensions[0] ;
    for (std::size_t i =1 ; i < v.num_repeated_dimensions; ++i) {
      std::cout << ", "<< v.repeated_dimensions[i];
    }
    std::cout << "]\n";
    std::cout << "num_non_repeated_dimensions: "
              << v.num_non_repeated_dimensions << '\n';
    std::cout << "non_repeated_dimensions: [" << v.non_repeated_dimensions[0] ;
    for (std::size_t i =1 ; i < v.num_non_repeated_dimensions; ++i) {
      std::cout << ", "<< v.non_repeated_dimensions[i];
    }
    std::cout << "]\n";

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
                   discrete_velocity.data().front().data());
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
                   discrete_velocity.data().front().data());
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
                   discrete_velocity.data().front().data());
    }


      //for (std::size_t z = 0; z < discrete_channelflow_domain.size<2>();++z){
      //for (std::size_t y = 0; y < discrete_channelflow_domain.size<1>();++y) {
      //  if (discrete_velocity(0, y, z).x() != 0 ||
      //      discrete_velocity(0, y, z).y() != 0 ||
      //      discrete_velocity(0, y, z).z() != 0) {
      //    std::cout << "boingsi\n";
      //  }
      //}
      //}

    indicator.set_text("Creating slabs for infinite domain");
    repeat_for_infinite<1, 2>(discrete_velocity);

    //--------------------------------------------------------------------------
    // indicator.set_text("Building numerical flowmap");
    auto phi = flowmap(v);
    phi.use_caching(false);
    //--------------------------------------------------------------------------
    indicator.set_text("Advecting path lines");
    auto pathlines = std::vector<line3> {};
    auto const bb = discrete_channelflow_domain.bounding_box();
    for (std::size_t i = 0; i < 100; ++i) {
      auto x0 = bb.random_point();
      x0.y()  = 0;

      pathlines.push_back(trace_flow(v, x0, args.t0, args.tau));
    }
    //--------------------------------------------------------------------------
    indicator.set_text("Writing path lines");
    write_vtk(pathlines, "channelflow_pathlines.vtk");
  });
}
