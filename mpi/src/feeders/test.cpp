#include <mpi.h>
#include <tatooine/axis_aligned_bounding_box.h>
#include <tatooine/mpi/feeders/analytical_function.h>
#include <tatooine/mpi/interfaces/test.h>

#include <boost/multi_array.hpp>
#include <boost/program_options.hpp>
#include <cmath>
#include <iostream>
#include <iterator>
//==============================================================================
namespace po = boost::program_options;
using namespace tatooine::mpi::feeders;
using scalar_array = boost::multi_array<double, 3>;
using vec_array    = boost::multi_array<double, 4>;
using index_t      = scalar_array::index;
using range_t      = scalar_array::extent_range;
using flow_t       = std::unique_ptr<ScalarFieldInFlow>;
//==============================================================================
int                nDims = 2;
std::array<int, 2> dims{0, 0};
std::array<int, 2> periods{0, 0};

int                zero = 0;
int                rank = 0;
int                size = 0;
std::array<int, 3> grid_size{0, 0, 0};
tatooine::aabb3    aabb;
int                restart          = 0;
int                iteration        = 0;
double             t0               = 0;
double             t1               = 0;
double             cur_t            = t0;
double             dt               = 0;
bool               use_interpolated = false;

int  endy          = 0;
int  endz          = 0;
bool isSingleCellY = true;
bool isSingleCellZ = true;

int rSizey = 0;
int rSizez = 0;

int rNperY = 0;
int rNperZ = 0;

std::array<int, 2> rDims{0, 0};
std::array<int, 2> rPeriods{0, 0};
std::array<int, 2> rCoords{0, 0};

int starty = 0;
int startz = 0;

int gridstx = 0;
int gridsty = 0;
int gridstz = 0;

double xst = 0;
double yst = 0;
double zst = 0;

int periodicx = 0;
int periodicy = 0;
int periodicz = 0;

double dx = 0;
double dy = 0;
double dz = 0;

double deltaX = 0;
double deltaY = 0;
double deltaZ = 0;

int halo_level = 4;

std::unique_ptr<vec_array> velocity_field;
flow_t                     flow;
//==============================================================================
auto initialize_mpi(int argc, char** argv) -> void {
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // My ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);  // Number of Processes
  if (rank == 0) {
    std::cerr << "Initializing MPI Environment.\n";
  }
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
      if (rank == 0) {
        std::cerr << desc << "\n";
      }
      return false;
    }

    po::notify(vm);
    grid_size        = {vm["gridsizex"].as<int>(), vm["gridsizey"].as<int>(),
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

    if (rank == 0) {
      std::cerr << "grid size x was set to " << grid_size[0] << ".\n";
      std::cerr << "grid size y was set to " << grid_size[1] << ".\n";
      std::cerr << "grid size z was set to " << grid_size[2] << ".\n";

      std::cerr << "t0 was set to " << t0 << ".\n";
      std::cerr << "t1 was set to " << t1 << ".\n";
      std::cerr << "dt was set to " << dt << ".\n";
    }

    if (vm.count("restart_iteration") > 0) {
      restart   = 1;
      iteration = vm["restart_iteration"].as<int>();
      if (rank == 0) std::cerr << "restarting from step " << iteration << '\n';
      t0 = t0 + iteration * dt;
    }

  } catch (std::exception& e) {
    if (rank == 0) {
      std::cerr << "error: " << e.what() << "\n";
    }
    return false;
  } catch (...) {
    if (rank == 0) {
      std::cerr << "Exception of unknown type!\n";
    }
  }
  return true;
}
//------------------------------------------------------------------------------
auto simulation_step(ScalarFieldInFlow const& flow) -> void {
  if (rank == 0) {
    std::cerr << "Loop interation: " << iteration << ".\n";
  }
  cur_t = cur_t + dt;
  // Build new field
  for (auto k = index_t{startz}; k < endz; ++k) {
    for (auto j = index_t{starty}; j < endy; ++j) {
      for (auto i = index_t{0}; i < grid_size[0]; ++i) {
        auto const x                  = xst + double(i) * deltaX;
        auto const y                  = yst + double(j - starty) * deltaY;
        auto const z                  = zst + double(k - startz) * deltaZ;
        auto const vel                = flow.v(x, y, z, t0);
        (*velocity_field)[i][j][k][0] = vel.x();
        (*velocity_field)[i][j][k][1] = vel.y();
        (*velocity_field)[i][j][k][2] = vel.z();
      }
    }
  }
  if (rank == 0) {
    std::cerr << "Made new Field.\n";
  }
  tatooine_mpi_test_update_variables(velocity_field->data());
  if (rank == 0) {
    std::cerr << "Variables updated.\n";
  }
  tatooine_mpi_test_update(&iteration, &cur_t);
  if (rank == 0) {
    std::cerr << "Tracking updated.\n";
  }

  ++iteration;
}
//------------------------------------------------------------------------------
auto simulation_loop(ScalarFieldInFlow const& flow) -> void {
  iteration++;
  while (cur_t <= t1) {
    simulation_step(flow);
  }
}
//------------------------------------------------------------------------------
auto calculate_grid_position_for_worker() -> void {
  isSingleCellY = (rDims[0] == 1);
  isSingleCellZ = (rDims[1] == 1);

  rNperY = int(std::floor(grid_size[1] / rDims[0]));
  rNperZ = int(std::floor(grid_size[2] / rDims[1]));

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
    endy = grid_size[1];
  } else {
    endy = starty + rNperY;
  }

  if (rCoords[1] == rDims[1] - 1) {
    endz = grid_size[2];
  } else {
    endz = startz + rNperZ;
  }

  std::cerr << rank << ": " << endy - starty << " , " << endz - startz
            << " <- number of points.\n";
  std::cerr << rank << ": " << starty << " , " << endy << " <- Range Y.\n";
  std::cerr << rank << ": " << startz << " , " << endz << " <- Range Z.\n";

  gridstx = 0;
  gridsty = starty;
  gridstz = startz;
  rSizey  = endy - starty;
  rSizez  = endz - startz;

  // TODO No larger array when isSingleCell
  if (!isSingleCellY) {
    starty -= halo_level;
    endy += halo_level;
  }
  if (!isSingleCellZ) {
    startz -= halo_level;
    endz += halo_level;
  }
}
//------------------------------------------------------------------------------
auto create_flow() -> void {
  flow = flow_t{new TileBox{2 * M_PI, 0.0}};
  if (use_interpolated) {
    if (rank == 0) {
      std::cerr << "Using interpolated dataset\n";
    }
    flow = flow_t{new InterpolatedField{
        std::move(flow), std::set<double>{-1.0, 0.0, 1.0, 2.0, 3.0}}};
  }
}
//------------------------------------------------------------------------------
auto initialize_flow_data() {
  if (rank == 0) {
    std::cerr << "Filling the grid by function.\n";
  }

  create_flow();

  dx = aabb.max(0) - aabb.min(0);
  dy = aabb.max(1) - aabb.min(1);
  dz = aabb.max(2) - aabb.min(2);

  deltaX = periodicx ? dx / grid_size[0] : dx / (grid_size[0] - 1);
  deltaY = periodicy ? dy / grid_size[1] : dy / (grid_size[1] - 1);
  deltaZ = periodicz ? dz / grid_size[2] : dz / (grid_size[2] - 1);

  xst = aabb.min(0);
  yst = aabb.min(1) + starty * deltaY;
  zst = aabb.min(2) + startz * deltaZ;

  for (auto k = index_t{startz}; k < endz; ++k) {
    for (auto j = index_t{starty}; j < endy; ++j) {
      for (auto i = index_t{0}; i < grid_size[0]; ++i) {
        auto const x                  = xst + double(i) * deltaX;
        auto const y                  = yst + double(j - starty) * deltaY;
        auto const z                  = zst + double(k - startz) * deltaZ;
        auto const vel                = flow->v(x, y, z, t0);
        (*velocity_field)[i][j][k][0] = vel.x();
        (*velocity_field)[i][j][k][1] = vel.y();
        (*velocity_field)[i][j][k][2] = vel.z();
      }
    }
  }
  if (rank == 0) {
    std::cerr << "Done Filling grid.\n";
  }
}
//------------------------------------------------------------------------------
auto call_interface(MPI_Fint& commF) -> void {
  tatooine_mpi_test_initialize_communicator(&commF);
  if (rank == 0) {
    std::cerr << "Initialized MPI.\n";
  }

  tatooine_mpi_test_initialize_grid(
      &zero, &zero, &zero, &grid_size[0], &grid_size[1], &grid_size[2],
      &gridstx, &gridsty, &gridstz, &grid_size[0], &rSizey, &rSizez, &dx, &dy,
      &dz, &periodicx, &periodicy, &periodicz, &halo_level);
  if (rank == 0) {
    std::cerr << "Initialized grid.\n";
  }

  tatooine_mpi_test_initialize_variables(velocity_field->data());
  if (rank == 0) {
    std::cerr << "Initialized Variables.\n";
  }

  auto prev_time = t0 - dt;
  tatooine_mpi_test_initialize_parameters(&t0, &prev_time, &iteration);
  if (rank == 0) {
    std::cerr << "Initialized Parameters.\n";
  }

  tatooine_mpi_test_initialize(&restart);

  simulation_loop(*flow);
}
//==============================================================================
auto main(int argc, char** argv) -> int {
  initialize_mpi(argc, argv);
  if (!parse_args(argc, argv)) {
    return 0;
  }

  // Define Cartesian Topology
  if (rank == 0) {
    std::cerr << "Create Cartesian Topology.\n";
  }
  MPI_Comm newComm;

  MPI_Dims_create(size, nDims, dims.data());
  MPI_Cart_create(MPI_COMM_WORLD, nDims, dims.data(), periods.data(), true,
                  &newComm);

  MPI_Comm_set_errhandler(newComm, MPI_ERRORS_RETURN);

  // Allocate required space
  if (rank == 0) {
    std::cerr << "Get dimension and position.\n";
  }
  MPI_Cartdim_get(newComm, &nDims);

  // Get Position in Cartesian grid
  MPI_Cart_get(newComm, nDims, rDims.data(), rPeriods.data(), rCoords.data());
  std::cerr << rank << ": " << rDims[0] << " , " << rDims[1]
            << " <- Dimensions \n";
  std::cerr << rank << ": " << rCoords[0] << " , " << rCoords[1]
            << " <- Rank Coordinate.\n";

  // Convert to FInt
  MPI_Fint commF = MPI_Comm_c2f(newComm);

  calculate_grid_position_for_worker();
  velocity_field = std::make_unique<vec_array>(
      boost::extents[range_t(0, grid_size[0])][range_t(starty, endy)]
                    [range_t(startz, endz)][3],
      boost::fortran_storage_order());

  initialize_flow_data();

  call_interface(commF);
  MPI_Finalize();
  return 0;
}
