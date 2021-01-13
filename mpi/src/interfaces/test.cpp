#include <tatooine/mpi/interfaces/base_interface.h>
#include <tatooine/mpi/interfaces/test.h>

//==============================================================================
namespace tatooine::mpi::interfaces {
//==============================================================================
struct test : base_interface<test> {
  static constexpr std::string_view m_timings_fname     = "test_timings.txt";
  static constexpr std::string_view m_memory_fname      = "test_memory.txt";
  static constexpr std::string_view m_split_merge_fname = "test_splitmerge.txt";

  static constexpr std::string_view m_output_dir_name = "data_test";

  double   m_time;
  double   m_prev_time;
  uint64_t m_iteration;

  bool m_grid_initialized       = false;
  bool m_variables_initialized  = false;
  bool m_parameters_initialized = false;
  bool m_initialized            = false;

  std::ofstream m_timings_file;

  //==============================================================================
  // Interface Functions
  //==============================================================================
  auto initialize_grid(int gridstx, int gridsty, int gridstz, int gridx,
                       int gridy, int gridz, int xst, int yst, int zst, int xsz,
                       int ysz, int zsz, double dx, double dy, double dz,
                       int /*periodicx*/, int /*periodicy*/, int /*periodicz*/,
                       int halo_level) -> void {
    if (m_grid_initialized) {
      return;
    }
    if (!m_mpi_communicator_initialized) {
      throw std::logic_error(
          "initialize_grid must be called after "
          "initialize");
    }
    log("Initializing grid");

    assert(gridx >= 0);
    assert(gridy >= 0);
    assert(gridz >= 0);
    assert(xsz >= 0);
    assert(ysz >= 0);
    assert(zsz >= 0);
    assert(dx >= 0);
    assert(dy >= 0);
    assert(dz >= 0);
    assert(halo_level >= 0 && halo_level <= UINT8_MAX);

    log_all("gridstx: " + std::to_string(gridstx));
    log_all("gridsty: " + std::to_string(gridsty));
    log_all("gridstz: " + std::to_string(gridstz));
    log_all("gridx: " + std::to_string(gridx));
    log_all("gridy: " + std::to_string(gridy));
    log_all("gridz: " + std::to_string(gridz));
    log_all("xst: " + std::to_string(xst));
    log_all("yst: " + std::to_string(yst));
    log_all("zst: " + std::to_string(zst));
    log_all("dx: " + std::to_string(dx));
    log_all("dy: " + std::to_string(dy));
    log_all("dz: " + std::to_string(dz));
    log_all("halo_level: " + std::to_string(halo_level));

    if (halo_level < 4) {
      throw std::invalid_argument("halo_level must be at least 4. Given: " +
                                  std::to_string(halo_level));
    }

    // auto to_BC = [](int* periodic) {
    //  return *periodic ? BC::periodic : BC::nonperiodic;
    //};
    // auto bc =
    //    BoundaryConditions(to_BC(periodicx), to_BC(periodicx),
    //    to_BC(periodicy),
    //                       to_BC(periodicy), to_BC(periodicz),
    //                       to_BC(periodicz));
    //
    // auto domain = DomainExtent{{*dx, *dy, *dz}};

    // Make boundary conditions for singleton dimensions periodic and make sure
    // their domain is zero-width to make code simpler
    // if (*gridx == 1) {
    //  bc[Direction::left] = bc[Direction::right] = BC::periodic;
    //  domain.size[0]                             = 0;
    //}
    // if (*gridy == 1) {
    //  bc[Direction::down] = bc[Direction::up] = BC::periodic;
    //  domain.size[1]                          = 0;
    //}
    // if (*gridz == 1) {
    //  bc[Direction::front] = bc[Direction::back] = BC::periodic;
    //  domain.size[2]                             = 0;
    //}

    // two ghost particles for each non-singleton dimension minus one
    // auto grid = UniformGrid{
    //    GridExtent{{*gridstx, *gridsty, *gridstz}, {*gridx, *gridy, *gridz}},
    //    domain, bc};
    //
    // auto local_extent = GridExtent{{*xst, *yst, *zst}, {*xsz, *ysz, *zsz}};
    //
    //
    //_min_init_distance = grid.deltas().norm() * 0.33;
    m_grid_initialized = true;
  }
  //------------------------------------------------------------------------------
  auto initialize_variables(double* /*flow_velocity*/) -> void {
    if (m_variables_initialized) {
      return;
    }
    if (!m_grid_initialized) {
      throw std::logic_error(
          "initialize_variables must be called "
          "after initialize_grid");
    }
    log("Initializing variables");
    // Variables
    m_variables_initialized = true;
  }
  //------------------------------------------------------------------------------
  auto initialize_parameters(double time, double prev_time, int iteration)
      -> void {
    if (m_parameters_initialized) return;
    if (!m_grid_initialized) {
      throw std::logic_error(
          "initialize_parameters must be called "
          "after initialize_grid");
    }
    log("Initializing parameters");
    m_time      = time;
    m_prev_time = prev_time;
    m_iteration = iteration;

    // the local blocks must be large enough so that ghost particles never
    // wander further than into the neighboring processor
    m_parameters_initialized = true;
  }
  //------------------------------------------------------------------------------
  auto initialize(int restart) -> void {
    base_interface<test>::initialize_memory_file(restart, m_memory_fname);
    if (m_initialized) {
      return;
    }
    if (!m_variables_initialized || !m_parameters_initialized) {
      throw std::logic_error(
          "initialize must be called "
          "after initialize_parameters and "
          "initialize_variables");
    }
    log("Initializing");

    m_base_pmused = pm_used();
    m_base_vmused = vm_used();

    // create output directory
    std::filesystem::create_directories(m_output_dir_name);

    if (restart == 1) {
      // Append to log files
      m_timings_file.open(std::filesystem::path{m_timings_fname},
                          std::ios::app);
    } else {
      // Clear contents of log files
      m_timings_file.open(std::filesystem::path{m_timings_fname},
                          std::ios::trunc);
      // Seed particles on iso surface
    }
    m_initialized = true;
  }
  //------------------------------------------------------------------------------
  auto update_variables(double* /*flow_velocity*/) -> void {
    if (!m_initialized) {
      throw std::logic_error(
          "update_variables can only be called if "
          "initialization is complete");
    }
    log("Updating variables");
    // Here goes updating variables
  }
  //------------------------------------------------------------------------------
  auto update(int iteration, double time) -> void {
    if (!m_initialized) {
      throw std::logic_error(
          "update can only be called if "
          "initialization is complete");
    }
    m_prev_time = m_time;
    m_time      = time;
    m_iteration = iteration;
    log("Updating for iteration " + std::to_string(m_iteration));

    auto ct = std::chrono::system_clock::now();
    if (m_mpi_communicator->rank() == 0) {
      auto sim_time = ct - m_last_end_time;
      m_timings_file << m_iteration << '\t'
                     << std::chrono::duration_cast<std::chrono::milliseconds>(
                            sim_time)
                            .count()
                     << '\n';
    }

    // Here goes update

    log("Tatooine update step finished");
    //log_mem_usage(m_iteration);
    m_last_end_time = std::chrono::system_clock::now();
  }
};
//==============================================================================
}  // namespace tatooine::mpi::interfaces
//==============================================================================

//==============================================================================
// Interface Functions
//==============================================================================
auto tatooine_mpi_test_initialize_communicator(MPI_Fint* communicator) -> void {
  tatooine::mpi::interfaces::test::get().initialize_communicator(*communicator);
}
//------------------------------------------------------------------------------
auto tatooine_mpi_test_initialize_grid(int* gridstx, int* gridsty, int* gridstz,
                                       int* gridx, int* gridy, int* gridz,
                                       int* xst, int* yst, int* zst, int* xsz,
                                       int* ysz, int* zsz, double* dx,
                                       double* dy, double* dz, int* periodicx,
                                       int* periodicy, int* periodicz,
                                       int* halo_level) -> void {
  tatooine::mpi::interfaces::test::get().initialize_grid(
      *gridstx, *gridsty, *gridstz, *gridx, *gridy, *gridz, *xst, *yst, *zst,
      *xsz, *ysz, *zsz, *dx, *dy, *dz, *periodicx, *periodicy, *periodicz,
      *halo_level);
}
//------------------------------------------------------------------------------
auto tatooine_mpi_test_initialize_variables(double* flow_velocity) -> void {
  tatooine::mpi::interfaces::test::get().initialize_variables(flow_velocity);
}
//------------------------------------------------------------------------------
auto tatooine_mpi_test_initialize_parameters(double* time, double* prev_time,
                                             int* iteration) -> void {
  tatooine::mpi::interfaces::test::get().initialize_parameters(
      *time, *prev_time, *iteration);
}
//------------------------------------------------------------------------------
auto tatooine_mpi_test_initialize(int* restart) -> void {
  tatooine::mpi::interfaces::test::get().initialize(*restart);
}
//------------------------------------------------------------------------------
auto tatooine_mpi_test_update_variables(double* flow_velocity) -> void {
  tatooine::mpi::interfaces::test::get().update_variables(flow_velocity);
}
//------------------------------------------------------------------------------
auto tatooine_mpi_test_update(int* iteration, double* time) -> void {
  tatooine::mpi::interfaces::test::get().update(*iteration, *time);
}
