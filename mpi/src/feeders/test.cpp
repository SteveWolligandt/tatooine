#include <mpi.h>
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
//==============================================================================
auto main(int argc, char** argv) -> int {
  // Initialize MPI Environment
  auto rank = 0;
  auto size = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // My ID
  MPI_Comm_size(MPI_COMM_WORLD, &size);  // Number of Processes
  if (rank == 0) {
    std::cerr << "Initializing MPI Environment.\n";
  }

  auto gridsizex = int{};
  auto gridsizey = gridsizex;
  auto gridsizez = gridsizex;

  auto transformation_start_time    = 0.0;
  auto interval_step                = 0.0;
  auto num_transformation_intervals = int{0};
  auto split_interval               = int{1};
  auto saving_tracers_interval      = int{1};
  auto restart                      = int{0};
  auto iteration                    = int{0};
  auto max_init_distance            = 0.0;
  auto seeding_error_init           = 0.01;
  auto normal_error_max             = 0.0;
  auto normal_error_min             = 0.0;
  auto tangential_error_max         = 0.0;
  auto tangential_error_min         = 0.0;
  auto target_iso_value             = 0.0;
  auto x0                           = 0.0;
  auto x1                           = 0.0;
  auto y0                           = 0.0;
  auto y1                           = 0.0;
  auto z0                           = 0.0;
  auto z1                           = 0.0;
  auto t0                           = 0.0;
  auto t1                           = 0.0;
  auto dt                           = 0.0;
  auto use_interpolated             = false;
  try {
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")(
        "gridsizex", po::value<int>()->required(),
        "Number of grid points in x direction")(
        "gridsizey", po::value<int>()->required(),
        "Number of grid points in y direction")(
        "gridsizez", po::value<int>()->required(),
        "Number of grid points in z direction")(
        "split_interval", po::value<int>()->default_value(1),
        "Number of iterations between checking for splitting tracers")(
        "saving_tracers_interval", po::value<int>()->required(),
        "Number of iterations between saving tracer states")(
        "transformation_start_time", po::value<double>()->default_value(0.0),
        "Number of iterations between saving tracer states")(
        "interval_step", po::value<double>()->default_value(0.0),
        "Number of iterations between saving tracer states")(
        "num_transformation_intervals", po::value<int>()->default_value(0),
        "Number of iterations between saving tracer states")(
        "max_init_distance", po::value<double>()->required(),
        "Maximum radius of a circle on the surface that contains no tracers")(
        "seeding_error_init",
        po::value<double>()->required()->default_value(0.01),
        "Maximum deviation from a plane (relative to curvature radius) in the "
        "neighborhood when considering adding or removing tracers")(
        "normal_error_max", po::value<double>()->required(),
        "Maximum error in normal direction before splitting")(
        "normal_error_min", po::value<double>()->required(),
        "Maximum error in normal direction before splitting")(
        "tangential_error_max", po::value<double>()->required(),
        "Maximum error in normal direction before splitting")(
        "tangential_error_min", po::value<double>()->required(),
        "Maximum error in normal direction before splitting")(
        "target_iso_value", po::value<double>()->required()->default_value(0),
        "Iso value of the implicit surface to track")(
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
      return 0;
    }

    po::notify(vm);

    gridsizex                    = vm["gridsizex"].as<int>();
    gridsizey                    = vm["gridsizey"].as<int>();
    gridsizez                    = vm["gridsizez"].as<int>();
    transformation_start_time    = vm["transformation_start_time"].as<double>();
    interval_step                = vm["interval_step"].as<double>();
    num_transformation_intervals = vm["num_transformation_intervals"].as<int>();
    split_interval               = vm["split_interval"].as<int>();
    saving_tracers_interval      = vm["saving_tracers_interval"].as<int>();
    max_init_distance            = vm["max_init_distance"].as<double>();
    seeding_error_init           = vm["seeding_error_init"].as<double>();
    normal_error_max             = vm["normal_error_max"].as<double>();
    normal_error_min             = vm["normal_error_min"].as<double>();
    tangential_error_max         = vm["tangential_error_max"].as<double>();
    tangential_error_min         = vm["tangential_error_min"].as<double>();
    target_iso_value             = vm["target_iso_value"].as<double>();
    x0                           = vm["x0"].as<double>();
    x1                           = vm["x1"].as<double>();
    y0                           = vm["y0"].as<double>();
    y1                           = vm["y1"].as<double>();
    z0                           = vm["z0"].as<double>();
    z1                           = vm["z1"].as<double>();
    t0                           = vm["t0"].as<double>();
    t1                           = vm["t1"].as<double>();
    dt                           = vm["dt"].as<double>();
    use_interpolated             = vm["use_interpolated"].as<bool>();

    if (rank == 0) {
      std::cerr << "grid size x was set to " << gridsizex << ".\n";
      std::cerr << "grid size y was set to " << gridsizey << ".\n";
      std::cerr << "grid size z was set to " << gridsizez << ".\n";

      std::cerr << "saving_tracers_interval was set to "
                << saving_tracers_interval << ".\n";
      std::cerr << "max_init_distance " << max_init_distance << ".\n";

      std::cerr << "seeding_error_init was set to " << seeding_error_init
                << ".\n";

      std::cerr << "target_iso_value was set to " << target_iso_value << ".\n";
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
    return 1;
  } catch (...) {
    if (rank == 0) {
      std::cerr << "Exception of unknown type!\n";
    }
  }

  // Define Cartesian Topology
  if (rank == 0) {
    std::cerr << "Create Cartesian Topology.\n";
  }
  // int		*rDims, *rPeriods, *rCoords;
  auto     nDims   = 2;
  auto     dims    = std::array<int, 2>{0, 0};
  auto     periods = std::array<int, 2>{0, 0};
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
  auto rDims    = std::array<int, 2>{0, 0};
  auto rPeriods = std::array<int, 2>{0, 0};
  auto rCoords  = std::array<int, 2>{0, 0};

  // Get Position in Cartesian grid
  MPI_Cart_get(newComm, nDims, rDims.data(), rPeriods.data(), rCoords.data());
  std::cerr << rank << ": " << rDims[0] << " , " << rDims[1]
            << " <- Dimensions \n";
  std::cerr << rank << ": " << rCoords[0] << " , " << rCoords[1]
            << " <- Rank Coordinate.\n";

  // Convert to FInt
  MPI_Fint commF = MPI_Comm_c2f(newComm);

  // Calculate grid position foreach worker
  auto endy          = 0;
  auto endz          = 0;
  auto rSizey        = 0;
  auto rSizez        = 0;
  auto isSingleCellY = (rDims[0] == 1);
  auto isSingleCellZ = (rDims[1] == 1);

  auto rNperY = int(std::floor(gridsizey / rDims[0]));
  auto rNperZ = int(std::floor(gridsizez / rDims[1]));

  // add additional halo grid points
  // # of grid points is:
  // rNperY + 2*haloLevel
  // rNperZ + 2*haloLevel
  // Unless it's a single cell, where it's not added

  // when the cell is not a border cell, it just starts earlier and ends later.
  // If it's either the last or first cell, it either uses periodic information
  // or it ignores the values given in the halo position

  // start index
  auto starty = rNperY * rCoords[0];
  auto startz = rNperZ * rCoords[1];

  // end index changes when it's the last cell
  if (rCoords[0] == rDims[0] - 1) {
    endy = gridsizey;
  } else {
    endy = starty + rNperY;
  }

  if (rCoords[1] == rDims[1] - 1) {
    endz = gridsizez;
  } else {
    endz = startz + rNperZ;
  }

  std::cerr << rank << ": " << endy - starty << " , " << endz - startz
            << " <- number of points.\n";
  std::cerr << rank << ": " << starty << " , " << endy << " <- Range Y.\n";
  std::cerr << rank << ": " << startz << " , " << endz << " <- Range Z.\n";

  auto halo_level = 4;

  auto gridstx = 0;
  auto gridsty = starty;
  auto gridstz = startz;
  rSizey       = endy - starty;
  rSizez       = endz - startz;

  // No larger array when isSingleCell TODO
  if (!isSingleCellY) {
    starty -= halo_level;
    endy += halo_level;
  }
  if (!isSingleCellZ) {
    startz -= halo_level;
    endz += halo_level;
  }

  using scalar_array = boost::multi_array<double, 3>;
  using vec_array    = boost::multi_array<double, 4>;
  using index        = scalar_array::index;
  using range        = scalar_array::extent_range;
  auto sField        = scalar_array(boost::extents[range(0, gridsizex)][range(
                                 starty, endy)][range(startz, endz)],
                             boost::fortran_storage_order());
  auto gradField     = vec_array(boost::extents[range(0, gridsizex)][range(
                                 starty, endy)][range(startz, endz)][3],
                             boost::fortran_storage_order());
  auto newsField     = scalar_array(boost::extents[range(0, gridsizex)][range(
                                    starty, endy)][range(startz, endz)],
                                boost::fortran_storage_order());
  auto newgradField  = vec_array(boost::extents[range(0, gridsizex)][range(
                                    starty, endy)][range(startz, endz)][3],
                                boost::fortran_storage_order());
  auto vField        = vec_array(boost::extents[range(0, gridsizex)][range(
                              starty, endy)][range(startz, endz)][3],
                          boost::fortran_storage_order());

  if (rank == 0) {
    std::cerr << "Filling the grid by function.\n";
  }
  auto flow = std::unique_ptr<ScalarFieldInFlow>(new TileBox(2 * M_PI, 0.0));
  // auto flow = unique_ptr<ScalarFieldInFlow>(new Plane());

  if (use_interpolated) {
    if (rank == 0) {
      std::cerr << "Using interpolated dataset\n";
    }
    flow = std::unique_ptr<ScalarFieldInFlow>(new InterpolatedField{
        std::move(flow), std::set<double>{-1.0, 0.0, 1.0, 2.0, 3.0}});
  }

  auto prev_time = t0 - dt;

  auto periodicx = 0;
  auto periodicy = 0;
  auto periodicz = 0;

  auto dx = x1 - x0;
  auto dy = y1 - y0;
  auto dz = z1 - z0;

  auto deltaX = periodicx ? dx / gridsizex : dx / (gridsizex - 1);
  auto deltaY = periodicy ? dy / gridsizey : dy / (gridsizey - 1);
  auto deltaZ = periodicz ? dz / gridsizez : dz / (gridsizez - 1);

  auto xst = x0;
  auto yst = y0 + starty * deltaY;
  auto zst = z0 + startz * deltaZ;

  for (auto k = index{startz}; k < endz; ++k) {
    for (auto j = index{starty}; j < endy; ++j) {
      for (auto i = index{0}; i < gridsizex; ++i) {
        auto x                = xst + double(i) * deltaX;
        auto y                = yst + double(j - starty) * deltaY;
        auto z                = zst + double(k - startz) * deltaZ;
        sField[i][j][k]       = flow->s(x, y, z, prev_time);
        newsField[i][j][k]    = flow->s(x, y, z, t0);
        auto grad             = flow->g(x, y, z, t0);
        gradField[i][j][k][0] = grad.x();
        gradField[i][j][k][1] = grad.y();
        gradField[i][j][k][2] = grad.z();
        auto vel              = flow->v(x, y, z, t0);
        vField[i][j][k][0]    = vel.x();
        vField[i][j][k][1]    = vel.y();
        vField[i][j][k][2]    = vel.z();
      }
    }
  }
  if (rank == 0) {
    std::cerr << "Done Filling grid.\n";
  }

  tatooine::mpi::interfaces::test::tatooine_mpi_initialize(&commF);
  if (rank == 0) {
    std::cerr << "Initialized MPI.\n";
  }

  auto zero = 0;

  tatooine::mpi::interfaces::test::tatooine_mpi_initialize_grid(
      &zero, &zero, &zero, &gridsizex, &gridsizey, &gridsizez, &gridstx,
      &gridsty, &gridstz, &gridsizex, &rSizey, &rSizez, &dx, &dy, &dz,
      &periodicx, &periodicy, &periodicz, &halo_level);
  if (rank == 0) {
    std::cerr << "Initialized grid.\n";
  }

  tatooine::mpi::interfaces::test::tatooine_mpi_initialize_variables(
      newsField.data(), sField.data(), gradField.data(), vField.data(),
      sField.data());
  if (rank == 0) {
    std::cerr << "Initialized Variables.\n";
  }

  auto time = t0;

  tatooine::mpi::interfaces::test::tatooine_mpi_initialize_parameters(
      &max_init_distance, &seeding_error_init, &normal_error_max,
      &normal_error_min, &tangential_error_max, &tangential_error_min,
      &target_iso_value, &split_interval, &saving_tracers_interval,
      &transformation_start_time, &interval_step, &num_transformation_intervals,
      &t0, &prev_time, &iteration);
  if (rank == 0) {
    std::cerr << "Initialized Parameters.\n";
  }

  tatooine::mpi::interfaces::test::tatooine_mpi_initialize_tracking(&restart);

  // Update Loop:
  ++iteration;

  while (time <= t1) {
    if (rank == 0) {
      std::cerr << "Loop interation: " << iteration << ".\n";
    }
    time = time + dt;
    // Build new field
    for (auto k = index{startz}; k < endz; ++k) {
      for (auto j = index{starty}; j < endy; ++j) {
        for (auto i = index{0}; i < gridsizex; ++i) {
          auto x                   = xst + double(i) * deltaX;
          auto y                   = yst + double(j - starty) * deltaY;
          auto z                   = zst + double(k - startz) * deltaZ;
          newsField[i][j][k]       = flow->s(x, y, z, time);
          auto grad                = flow->g(x, y, z, time);
          newgradField[i][j][k][0] = grad.x();
          newgradField[i][j][k][1] = grad.y();
          newgradField[i][j][k][2] = grad.z();
        }
      }
    }
    if (rank == 0) {
      std::cerr << "Made new Field.\n";
    }
    tatooine::mpi::interfaces::test::tatooine_mpi_update_variables(
        newsField.data(), sField.data(), newgradField.data(), vField.data(),
        sField.data());
    if (rank == 0) {
      std::cerr << "Variables updated.\n";
    }
    tatooine::mpi::interfaces::test::tatooine_mpi_update(&iteration, &time);
    if (rank == 0) {
      std::cerr << "Tracking updated.\n";
    }
    sField    = newsField;
    gradField = newgradField;

    ++iteration;
  }

  MPI_Finalize();
  return 0;
}
