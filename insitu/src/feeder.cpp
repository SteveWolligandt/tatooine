#include <tatooine/mpi/program.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
#include <tatooine/analytical/fields/numerical/tornado.h>
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/insitu/c_interface.h>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
//==============================================================================
namespace po       = boost::program_options;
using scalar_array = boost::multi_array<double, 3>;
using vec_array    = boost::multi_array<double, 4>;
using index_t      = scalar_array::index;
using range_t      = scalar_array::extent_range;
//==============================================================================
int start_y, start_z, end_y, end_z;
std::array<double, 3> local_domain_origin;
std::array<double, 3> domain_size;
std::array<double, 3> delta;

[[maybe_unused]] int const zero  = 0;
[[maybe_unused]] int const one   = 1;
[[maybe_unused]] int const two   = 2;
[[maybe_unused]] int const three = 3;
std::array<int, 3>         global_grid_size{0, 0, 0};
tatooine::aabb3            aabb;
int                        restart          = 0;
int                        iteration        = 0;
double                     t0               = 0;
double                     t1               = 0;
double                     cur_t            = t0;
double                     dt               = 0;
bool                       use_interpolated = false;


int halo_level = 1;

std::unique_ptr<scalar_array> velocity_x_field, velocity_y_field,
    velocity_z_field;
//==============================================================================
auto parse_args(int argc, char** argv) {
  try {
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "gridsizex", po::value<int>()->required(),
        "Number of grid points in x direction")(
        "gridsizey", po::value<int>()->required(),
        "Number of grid points in y direction")(
        "gridsizez", po::value<int>()->required(),
        "Number of grid points in z direction")(
        "x0", po::value<double>()->default_value(0), "start of x bounding box")(
        "x1", po::value<double>()->default_value(3), "end of x bounding box")(
        "y0", po::value<double>()->default_value(0), "start of y bounding box")(
        "y1", po::value<double>()->default_value(1), "end of y bounding box")(
        "z0", po::value<double>()->default_value(-0.5),
        "start of z bounding box")(
        "z1", po::value<double>()->default_value(0.5), "end of z bounding box")(
        "t0", po::value<double>()->required()->default_value(0),
        "Starting time")(
        "t1", po::value<double>()->required()->default_value(1.0), "End time")(
        "dt", po::value<double>()->required()->default_value(1e-3),
        "Time step")("restart_iteration,r", po::value<int>(),
                     "iteration number to restart from")(
        "use_interpolated,i",
        po::bool_switch()->implicit_value(true)->default_value(false),
        "should we use the interpolated or full analytic dataset?");

    auto vm = po::variables_map{};
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.empty() || vm.count("help")) {
      return false;
    }

    po::notify(vm);
    global_grid_size = {vm["gridsizex"].as<int>(),
                        vm["gridsizey"].as<int>(),
                        vm["gridsizez"].as<int>()};
    aabb.min(0)      = vm["x0"].as<double>();
    aabb.max(0)      = vm["x1"].as<double>();
    aabb.min(1)      = vm["y0"].as<double>();
    aabb.max(1)      = vm["y1"].as<double>();
    aabb.min(2)      = vm["z0"].as<double>();
    aabb.max(2)      = vm["z1"].as<double>();
    t0               = vm["t0"].as<double>();
    t1               = vm["t1"].as<double>();
    cur_t            = t0;
    dt               = vm["dt"].as<double>();
    use_interpolated = vm["use_interpolated"].as<bool>();

    if (vm.count("restart_iteration") > 0) {
      restart   = 1;
      iteration = vm["restart_iteration"].as<int>();
      t0 = t0 + iteration * dt;
    }

  } catch (std::exception& e) {
    return false;
  } catch (...) {
  }
  return true;
}
//------------------------------------------------------------------------------
auto sample_flow() {
  //tatooine::analytical::fields::numerical::abcflow v;
  tatooine::analytical::fields::numerical::tornado v;
  for (int i = 0; i < global_grid_size[0]; ++i) {
    for (int j = start_y; j < end_y; ++j) {
      for (int k = start_z; k < end_z; ++k) {
        auto const vel = v(tatooine::vec3{local_domain_origin[0] + double(i) * delta[0],
                                          local_domain_origin[1] + double(j - start_y) * delta[1],
                                          local_domain_origin[2] + double(k - start_z) * delta[2]},
                           cur_t);
        (*velocity_x_field)[i][j][k] = vel.x();
        (*velocity_y_field)[i][j][k] = vel.y();
        (*velocity_z_field)[i][j][k] = vel.z();
      }
    }
  }
}
//------------------------------------------------------------------------------
auto simulation_step() -> void {
  cur_t = cur_t + dt;
  sample_flow();
  tatooine_insitu_interface_update_velocity_x(velocity_x_field->data());
  tatooine_insitu_interface_update_velocity_y(velocity_y_field->data());
  tatooine_insitu_interface_update_velocity_z(velocity_z_field->data());
  tatooine_insitu_interface_update(&iteration, &cur_t);

  ++iteration;
}
//------------------------------------------------------------------------------
auto simulation_loop() -> void {
  iteration++;
  while (cur_t <= t1) {
    simulation_step();
  }
}
//------------------------------------------------------------------------------
auto start_simulation() -> void {

  tatooine_insitu_interface_initialize_velocity_x(velocity_x_field->data());
  tatooine_insitu_interface_initialize_velocity_y(velocity_y_field->data());
  tatooine_insitu_interface_initialize_velocity_z(velocity_z_field->data());

  auto const prev_time = t0 - dt;
  tatooine_insitu_interface_initialize_parameters(&t0, &prev_time, &iteration);
  tatooine_insitu_interface_initialize(&restart);

  simulation_loop();
}
//==============================================================================
auto main(int argc, char** argv) -> int {
  auto& mpi_prog = tatooine::mpi::program::get(argc, argv);
  if (!parse_args(argc, argv)) {
    return 0;
  }
  mpi_prog.init_communicator(global_grid_size[1], global_grid_size[2]);
  auto comm_fint = mpi_prog.communicator_fint();
  tatooine_insitu_interface_initialize_communicator(&comm_fint);

  if (mpi_prog.rank() == 0) {
    std::cout << mpi_prog.num_processes() << '\n';
  }

  start_y = mpi_prog.process_begin(0);
  start_z = mpi_prog.process_begin(1);
  end_y   = mpi_prog.process_end(0);
  end_z   = mpi_prog.process_end(1);
  if (!mpi_prog.is_single_cell(0))  {
    start_y -= halo_level;
    end_y += halo_level;
  }
  if (!mpi_prog.is_single_cell(1))  {
    start_z -= halo_level;
    end_z += halo_level;
  }
  int pb0 = mpi_prog.process_begin(0);
  int pb1 = mpi_prog.process_begin(1);
  int ps0 = mpi_prog.process_size(0);
  int ps1 = mpi_prog.process_size(1);
  domain_size[0] = aabb.max(0) - aabb.min(0);
  domain_size[1] = aabb.max(1) - aabb.min(1);
  domain_size[2] = aabb.max(2) - aabb.min(2);

  int is_periodic_x = mpi_prog.is_periodic(0);
  int is_periodic_y = mpi_prog.is_periodic(1);
  int is_periodic_z = mpi_prog.is_periodic(2);
  delta[0] = is_periodic_x ? domain_size[0] / global_grid_size[0]
                            : domain_size[0] / (global_grid_size[0] - 1);
  delta[1] = is_periodic_y ? domain_size[1] / global_grid_size[1]
                            : domain_size[1] / (global_grid_size[1] - 1);
  delta[2] = is_periodic_z ? domain_size[2] / global_grid_size[2]
                            : domain_size[2] / (global_grid_size[2] - 1);

  local_domain_origin[0] = aabb.min(0);
  local_domain_origin[1] = aabb.min(1) + start_y * delta[1];
  local_domain_origin[2] = aabb.min(2) + start_z * delta[2];

  auto const extents = boost::extents[range_t(0, global_grid_size[0])][range_t(
      start_y, end_y)][range_t(start_z, end_z)];
  velocity_x_field =
      std::make_unique<scalar_array>(extents, boost::fortran_storage_order());
  velocity_y_field =
      std::make_unique<scalar_array>(extents, boost::fortran_storage_order());
  velocity_z_field =
      std::make_unique<scalar_array>(extents, boost::fortran_storage_order());

  sample_flow();
  tatooine_insitu_interface_initialize_grid(
      &global_grid_size[0], &global_grid_size[1], &global_grid_size[2],
      &zero, &pb0, &pb1,
      &global_grid_size[0], &ps0, &ps1,
      &domain_size[0], &domain_size[1], &domain_size[2],
      &is_periodic_x, &is_periodic_y, &is_periodic_z,
      &halo_level);
  start_simulation();
  return 0;
}
