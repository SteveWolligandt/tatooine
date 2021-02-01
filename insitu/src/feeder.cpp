#include <mpi.h>
#include <tatooine/analytical/fields/numerical/abcflow.h>
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
int                num_dimensions = 2;
std::array<int, 2> dims{0, 0};
std::array<int, 2> periods{0, 0};

[[maybe_unused]] int const zero  = 0;
[[maybe_unused]] int const one   = 1;
[[maybe_unused]] int const two   = 2;
[[maybe_unused]] int const three = 3;
int                        rank  = 0;
int                        size  = 0;
std::array<int, 3>         global_grid_size{0, 0, 0};
tatooine::aabb3            aabb;
int                        restart          = 0;
int                        iteration        = 0;
double                     t0               = 0;
double                     t1               = 0;
double                     cur_t            = t0;
double                     dt               = 0;
bool                       use_interpolated = false;

int  local_grid_end_y          = 0;
int  local_grid_end_z          = 0;
bool is_single_cell_y = true;
bool is_single_cell_z = true;

int local_grid_size_y = 0;
int local_grid_size_z = 0;

int rNperY = 0;
int rNperZ = 0;

std::array<int, 2> rDims{0, 0};
std::array<int, 2> rPeriods{0, 0};
std::array<int, 2> rCoords{0, 0};

int starty = 0;
int startz = 0;

int local_starting_index_x = 0;
int local_starting_index_y = 0;
int local_starting_index_z = 0;

double local_domain_origin_x = 0;
double local_domain_origin_y = 0;
double local_domain_origin_z = 0;

int is_periodic_x = 0;
int is_periodic_y = 0;
int is_periodic_z = 0;

double domain_size_x = 0;
double domain_size_y = 0;
double domain_size_z = 0;

double deltaX = 0;
double deltaY = 0;
double deltaZ = 0;

int halo_level = 4;

std::unique_ptr<scalar_array> velocity_x_field, velocity_y_field,
    velocity_z_field;
//==============================================================================
auto initialize_mpi(int argc, char** argv) -> void {
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // My ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);  // Number of Processes
}
//------------------------------------------------------------------------------
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
    global_grid_size = {vm["gridsizex"].as<int>(), vm["gridsizey"].as<int>(),
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
  tatooine::analytical::fields::numerical::abcflow v;
  for (int i = 0; i < global_grid_size[0]; ++i) {
    for (int j = starty; j < local_grid_end_y; ++j) {
      for (int k = startz; k < local_grid_end_z; ++k) {
        auto const vel = v(tatooine::vec3{local_domain_origin_x + double(i) * deltaX,
                                          local_domain_origin_y + double(j - starty) * deltaY,
                                          local_domain_origin_z + double(k - startz) * deltaZ},
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
auto calculate_grid_position_for_worker() -> void {
  is_single_cell_y = (rDims[0] == 1);
  is_single_cell_z = (rDims[1] == 1);

  rNperY = int(std::floor(global_grid_size[1] / rDims[0]));
  rNperZ = int(std::floor(global_grid_size[2] / rDims[1]));

  // add additional halo grid points
  // # of grid points is:
  // rNperY + 2*haloLevel
  // rNperZ + 2*haloLevel
  // Unless it's a single cell, where it's not added

  // when the cell is not a border cell, it just starts earlier and ends later.
  // If it's either the last or first cell, it either uses periodic information
  // or it ignores the values given in the halo position

  // start index
  starty = rNperY * rCoords[0];
  startz = rNperZ * rCoords[1];

  // end index changes when it's the last cell
  if (rCoords[0] == rDims[0] - 1) {
    local_grid_end_y = global_grid_size[1];
  } else {
    local_grid_end_y = starty + rNperY;
  }

  if (rCoords[1] == rDims[1] - 1) {
    local_grid_end_z = global_grid_size[2];
  } else {
    local_grid_end_z = startz + rNperZ;
  }

  local_starting_index_x = 0;
  local_starting_index_y = starty;
  local_starting_index_z = startz;
  local_grid_size_y      = local_grid_end_y - starty;
  local_grid_size_z      = local_grid_end_z - startz;

  // TODO No larger array when isSingleCell
  if (!is_single_cell_y) {
    starty -= halo_level;
    local_grid_end_y += halo_level;
  }
  if (!is_single_cell_z) {
    startz -= halo_level;
    local_grid_end_z += halo_level;
  }
}
//------------------------------------------------------------------------------
auto initialize_flow_data() {

  domain_size_x = aabb.max(0) - aabb.min(0);
  domain_size_y = aabb.max(1) - aabb.min(1);
  domain_size_z = aabb.max(2) - aabb.min(2);

  deltaX = is_periodic_x ? domain_size_x / global_grid_size[0]
                         : domain_size_x / (global_grid_size[0] - 1);
  deltaY = is_periodic_y ? domain_size_y / global_grid_size[1]
                         : domain_size_y / (global_grid_size[1] - 1);
  deltaZ = is_periodic_z ? domain_size_z / global_grid_size[2]
                         : domain_size_z / (global_grid_size[2] - 1);

  local_domain_origin_x = aabb.min(0);
  local_domain_origin_y = aabb.min(1) + starty * deltaY;
  local_domain_origin_z = aabb.min(2) + startz * deltaZ;

  sample_flow();
}
//------------------------------------------------------------------------------
auto start_simulation() -> void {
  tatooine_insitu_interface_initialize_grid(
      &global_grid_size[0], &global_grid_size[1], &global_grid_size[2],
      &local_starting_index_x, &local_starting_index_y, &local_starting_index_z,
      &global_grid_size[0], &local_grid_size_y, &local_grid_size_z,
      &domain_size_x, &domain_size_y, &domain_size_z, &is_periodic_x,
      &is_periodic_y, &is_periodic_z, &halo_level);

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
  initialize_mpi(argc, argv);
  if (!parse_args(argc, argv)) {
    return 0;
  }

  MPI_Comm new_communicator;

  MPI_Dims_create(size, num_dimensions, dims.data());
  MPI_Cart_create(MPI_COMM_WORLD, num_dimensions, dims.data(), periods.data(), true,
                  &new_communicator);

  MPI_Comm_set_errhandler(new_communicator, MPI_ERRORS_RETURN);

  // Allocate required space
  MPI_Cartdim_get(new_communicator, &num_dimensions);

  // Get Position in Cartesian grid
  MPI_Cart_get(new_communicator, num_dimensions, rDims.data(), rPeriods.data(), rCoords.data());

  // Convert to FInt
  MPI_Fint comm_fint = MPI_Comm_c2f(new_communicator);

  calculate_grid_position_for_worker();
  velocity_x_field = std::make_unique<scalar_array>(
      boost::extents[range_t(0, global_grid_size[0])][range_t(
          starty, local_grid_end_y)][range_t(startz, local_grid_end_z)],
      boost::fortran_storage_order());
  velocity_y_field = std::make_unique<scalar_array>(
      boost::extents[range_t(0, global_grid_size[0])][range_t(
          starty, local_grid_end_y)][range_t(startz, local_grid_end_z)],
      boost::fortran_storage_order());
  velocity_z_field = std::make_unique<scalar_array>(
      boost::extents[range_t(0, global_grid_size[0])][range_t(
          starty, local_grid_end_y)][range_t(startz, local_grid_end_z)],
      boost::fortran_storage_order());

  initialize_flow_data();

  tatooine_insitu_interface_initialize_communicator(&comm_fint);

  start_simulation();
  MPI_Finalize();
  return 0;
}
