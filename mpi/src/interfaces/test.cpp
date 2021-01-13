#include <tatooine/mpi/interfaces/test.h>

#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
#include <filesystem>
#include <iomanip>

//==============================================================================
namespace tatooine::mpi::interfaces::test {
//==============================================================================
// Global Variables
//==============================================================================
constexpr auto _timings_fname     = "test_timings.txt";
constexpr auto _memory_fname      = "test_memory.txt";
constexpr auto _split_merge_fname = "test_splitmerge.txt";

constexpr auto _output_dir_name         = "data_test";
constexpr auto _max_tracers_per_process = 100000000000;

std::unique_ptr<boost::mpi::cartesian_communicator> _mpi_comm;

uint32_t _num_ghost_particles;
double   _tracer_halo_size;

double   _max_init_distance;
double   _min_init_distance;
double   _seeding_error_init;
double   _max_norm_error = 1000;
double   _min_norm_error = 0.00;
double   _max_tan_error  = 0.005;
double   _min_tan_error  = 0.00001;
uint32_t _split_interval;
double   _transformation_interval_step;
double   _transformation_start_time;
uint32_t _num_transformation_intervals;
double   _target_iso_value;
double   _time;
double   _prev_time;
uint64_t _iteration;

uint64_t _saving_interval;

uint64_t _transformation_interval_index;

bool _mpi_initialized        = false;
bool _grid_initialized       = false;
bool _variables_initialized  = false;
bool _parameters_initialized = false;
bool _tracking_initialized   = false;

long _base_vmused = 0;
long _base_pmused = 0;

std::chrono::time_point<std::chrono::system_clock> _last_end_time;

std::ofstream _timings_file;
std::ofstream _memory_file;
std::ofstream _split_merge_file;

//==============================================================================
// Helper Functions
//==============================================================================
void log(const std::string& message) {
  if (_mpi_comm->rank() == 0) {
    std::cerr << message << '\n';
  }
}
//------------------------------------------------------------------------------
void log_all(const std::string& message) {
  std::cerr << _mpi_comm->rank() << ": " << message << '\n';
}
//------------------------------------------------------------------------------
long parse_line(char* line) {
  // This assumes that a digit will be found and the line ends in " Kb".
  auto        i = strlen(line);
  const char* p = line;
  while (*p < '0' || *p > '9')
    p++;
  line[i - 3] = '\0';
  i           = std::atol(p);
  return long(i);
}
//------------------------------------------------------------------------------
long vm_used() {  // Note: this value is in KB!
  FILE* file   = fopen("/proc/self/status", "r");
  long  result = -1;
  char  line[128];

  while (fgets(line, 128, file) != NULL) {
    if (strncmp(line, "VmSize:", 7) == 0) {
      result = parse_line(line);
      break;
    }
  }
  fclose(file);
  return result;
}
//------------------------------------------------------------------------------
long pm_used() {  // Note: this value is in KB!
  FILE* file   = fopen("/proc/self/status", "r");
  long  result = -1;
  char  line[128];

  while (fgets(line, 128, file) != NULL) {
    if (strncmp(line, "VmRSS:", 6) == 0) {
      result = parse_line(line);
      break;
    }
  }
  fclose(file);
  return result;
}
//------------------------------------------------------------------------------
void log_mem_usage() {
  // auto virtualMemUsedProcess = vm_used();
  auto memUsedProcess     = pm_used();
  auto memOverheadProcess = memUsedProcess - _base_pmused;

  // Gather all memory usage at master node
  // auto vmused_all = std::vector<long>{};
  // boost::mpi::gather(*_mpi_comm, virtualMemUsedProcess, vmused_all, 0L);

  auto pmused_all = std::vector<long>{};
  boost::mpi::gather(*_mpi_comm, memUsedProcess, pmused_all, 0L);

  auto overhead_all = std::vector<long>{};
  boost::mpi::gather(*_mpi_comm, memOverheadProcess, overhead_all, 0L);

  // auto basevm_all = std::vector<long>{};
  // boost::mpi::gather(*_mpi_comm, _base_vmused, basevm_all, 0L);

  auto basepm_all = std::vector<long>{};
  boost::mpi::gather(*_mpi_comm, _base_pmused, basepm_all, 0L);

  if (_mpi_comm->rank() == 0) {
    // auto mima_vm = std::minmax_element(begin(vmused_all),
    //                                    end(vmused_all));
    // auto vm_total = std::accumulate(begin(vmused_all),
    //                                 end(vmused_all), 0LL);
    // auto vm_avg = vm_total / _mpi_comm->size();

    auto mima_pm  = std::minmax_element(begin(pmused_all), end(pmused_all));
    auto pm_total = std::accumulate(begin(pmused_all), end(pmused_all), 0LL);
    auto pm_avg   = pm_total / _mpi_comm->size();

    auto mima_overhead =
        std::minmax_element(begin(overhead_all), end(overhead_all));
    auto overhead_total =
        std::accumulate(begin(overhead_all), end(overhead_all), 0LL);
    auto overhead_avg = overhead_total / _mpi_comm->size();

    // auto mima_basevm = std::minmax_element(begin(basevm_all),
    //                                        end(basevm_all));
    // auto basevm_total = std::accumulate(begin(basevm_all),
    //                                     end(basevm_all), 0LL);
    // auto basevm_avg = basevm_total / _mpi_comm->size();

    auto mima_basepm = std::minmax_element(begin(basepm_all), end(basepm_all));
    auto basepm_total =
        std::accumulate(begin(basepm_all), end(basepm_all), 0LL);
    auto basepm_avg = basepm_total / _mpi_comm->size();

    // auto numtr_avg   = numtr_total / _mpi_comm->size();

    // std::cerr << "Virtual Memory: \n";
    // std::cerr << "======================================\n";
    // std::cerr << "Maximum per node: " << *(mima_vm.second) / 1024 << "Mb"
    //          << '\n';
    // std::cerr << "Minimum per node: " << *(mima_vm.first) / 1024 << "Mb\n";
    // std::cerr << "Average per node: " << vm_avg / 1024 << "Mb\n";
    // std::cerr << "Total: " << vm_total / 1024 << "Mb\n";
    // std::cerr << "Maximum baseline per node: " << *(mima_basevm.second) /
    // 1024
    //          << "Mb\n";
    // std::cerr << "Minimum
    //    baseline per node : " << *(mima_basevm.first)/1024 << " Mb\n ";
    //                        std::cerr
    //          << "Average baseline per node: "
    //          << basevm_avg / 1024
    //          << "Mb"
    //          << '\n';
    // std::cerr << "Total baseline: " << basevm_total / 1024 << "Mb\n";
    // std::cerr << '\n';

    std::cerr << "Physical Memory: \n";
    std::cerr << "======================================\n";
    std::cerr << "Maximum per node: " << *(mima_pm.second) / 1024 << "Mb"
              << '\n';
    std::cerr << "Minimum per node: " << *(mima_pm.first) / 1024 << "Mb"
              << '\n';
    std::cerr << "Average per node: " << pm_avg / 1024 << "Mb\n";
    std::cerr << "Total: " << pm_total / 1024 << "Mb\n";
    std::cerr << "Maximum baseline per node: " << *(mima_basepm.second) / 1024
              << "Mb\n";
    std::cerr << "Minimum baseline per node: " << *(mima_basepm.first) / 1024
              << "Mb\n";
    std::cerr << "Average baseline per node: " << basepm_avg / 1024 << "Mb"
              << '\n';
    std::cerr << "Total baseline: " << basepm_total / 1024 << "Mb\n";
    std::cerr << "Maximum overhead per node: " << *(mima_overhead.second) / 1024
              << "Mb\n";
    std::cerr << "Minimum overhead per node: " << *(mima_overhead.first) / 1024
              << "Mb\n";
    std::cerr << "Average overhead per node: " << overhead_avg / 1024 << "Mb"
              << '\n';
    std::cerr << "Total overhead: " << overhead_total / 1024 << "Mb" << '\n';
    std::cerr << '\n';

    _memory_file << _iteration
                 << '\t'
                 //<< numtr_total << '\t'
                 //<< *(mima_numtr.second) << '\t' << numtr_avg << '\t'
                 << pm_total / 1024 << '\t' << *(mima_pm.second) / 1024 << '\t'
                 << *(mima_pm.first) / 1024 << '\t' << pm_avg / 1024 << '\t'
                 << overhead_total / 1024 << '\t'
                 << *(mima_overhead.second) / 1024 << '\t'
                 << *(mima_overhead.first) / 1024 << '\t' << overhead_avg / 1024
                 << '\n';
  }
}
//==============================================================================
// Interface Functions
//==============================================================================
auto tatooine_mpi_initialize(MPI_Fint* communicator) -> void {
  if (_mpi_initialized) return;
  // Communicator should be the one describing the cartesian grid
  // of processors
  // Convert communicator to C
  _mpi_comm = std::unique_ptr<boost::mpi::cartesian_communicator>{
      new boost::mpi::cartesian_communicator{MPI_Comm_f2c(*communicator),
                                             boost::mpi::comm_attach}};
  log("Initializing MPI");
  _mpi_initialized = true;
}
//------------------------------------------------------------------------------
auto tatooine_mpi_initialize_grid(
    int* gridstx, int* gridsty, int* gridstz,
    int* gridx, int* gridy, int* gridz,
    int* xst, int* yst, int* zst,
    int* xsz, int* ysz, int* zsz,
    double* dx, double* dy, double* dz,
    int* periodicx, int* periodicy, int* periodicz,
    int* halo_level) -> void {
  if (_grid_initialized) return;
  if (!_mpi_initialized) {
    throw std::logic_error(
        "tatooine_mpi_initialize_grid must be called after "
        "tatooine_mpi_initialize");
  }
  log("Initializing grid");

  assert(*gridx >= 0);
  assert(*gridy >= 0);
  assert(*gridz >= 0);
  assert(*xsz >= 0);
  assert(*ysz >= 0);
  assert(*zsz >= 0);
  assert(*dx >= 0);
  assert(*dy >= 0);
  assert(*dz >= 0);
  assert(*halo_level >= 0 && *halo_level <= UINT8_MAX);

  log_all("gridstx: " + std::to_string(*gridstx));
  log_all("gridsty: " + std::to_string(*gridsty));
  log_all("gridstz: " + std::to_string(*gridstz));
  log_all("gridx: " + std::to_string(*gridx));
  log_all("gridy: " + std::to_string(*gridy));
  log_all("gridz: " + std::to_string(*gridz));
  log_all("xst: " + std::to_string(*xst));
  log_all("yst: " + std::to_string(*yst));
  log_all("zst: " + std::to_string(*zst));
  log_all("dx: " + std::to_string(*dx));
  log_all("dy: " + std::to_string(*dy));
  log_all("dz: " + std::to_string(*dz));
  log_all("halo_level: " + std::to_string(*halo_level));

  if (*halo_level < 4) {
    throw std::invalid_argument("halo_level must be at least 4. Given: " +
                                std::to_string(*halo_level));
  }

  // auto to_BC = [](int* periodic) {
  //  return *periodic ? BC::periodic : BC::nonperiodic;
  //};
  // auto bc =
  //    BoundaryConditions(to_BC(periodicx), to_BC(periodicx), to_BC(periodicy),
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
  _grid_initialized = true;
}
//------------------------------------------------------------------------------
auto tatooine_mpi_initialize_variables(double* mixture_fraction,
                                       double* prev_mixture_fraction,
                                       double* mixture_fraction_gradient,
                                       double* flow_velocity,
                                       double* heat_release) -> void {
  if (_variables_initialized) return;
  if (!_grid_initialized) {
    throw std::logic_error(
        "tatooine_mpi_initialize_variables must be called "
        "after tatooine_mpi_initialize_grid");
  }
  log("Initializing variables");
  // Variables
  _variables_initialized = true;
}
//------------------------------------------------------------------------------
auto tatooine_mpi_initialize_parameters(
    double* max_particle_distance, double* seeding_error_init,
    double* normal_error_max, double* normal_error_min,
    double* tangential_error_max, double* tangential_error_min,
    double* target_iso_value, int* split_interval, int* saving_tracers_interval,
    double* transformation_start_time, double* transformation_interval_step,
    int* num_transformation_intervals, double* time, double* prev_time,
    int* iteration) -> void {
  if (_parameters_initialized) return;
  if (!_grid_initialized) {
    throw std::logic_error(
        "tatooine_mpi_initialize_parameters must be called "
        "after tatooine_mpi_initialize_grid");
  }
  log("Initializing parameters");
  _max_init_distance            = *max_particle_distance;
  _seeding_error_init           = *seeding_error_init;
  _max_norm_error               = *normal_error_max;
  _min_norm_error               = *normal_error_min;
  _max_tan_error                = *tangential_error_max;
  _min_tan_error                = *tangential_error_min;
  _target_iso_value             = *target_iso_value;
  _split_interval               = *split_interval;
  _saving_interval              = *saving_tracers_interval;
  _transformation_start_time    = *transformation_start_time;
  _transformation_interval_step = *transformation_interval_step;
  _num_transformation_intervals = *num_transformation_intervals;
  _time                         = *time;
  _prev_time                    = *prev_time;
  _iteration                    = *iteration;

  _transformation_interval_index = uint64_t(std::floor(
      (_time - _transformation_start_time) / _transformation_interval_step));

  // @todo: if the ghost particle distances are not checked every time step,
  // they might wander out of the tracer halo before being reset.
  _tracer_halo_size = _max_init_distance * 1.5;

  // the local blocks must be large enough so that ghost particles never
  // wander further than into the neighboring processor
  if (*split_interval < 0) {
    throw std::invalid_argument("split_interval must be >= 0");
  }
  if (_transformation_start_time < 0) {
    throw std::invalid_argument("transformation_start_time must be >= 0");
  }
  if (_transformation_interval_step < 0) {
    throw std::invalid_argument("transformation_interval_step must be >= 0");
  }
  if (*num_transformation_intervals < 0) {
    throw std::invalid_argument("num_transformation_intervals must be >= 0");
  }
  _parameters_initialized = true;
}
//------------------------------------------------------------------------------
auto tatooine_mpi_initialize_tracking(int* restart) -> void {
  if (_tracking_initialized) {
    return;
  }
  if (!_variables_initialized || !_parameters_initialized) {
    throw std::logic_error(
        "tatooine_mpi_initialize_tracking must be called "
        "after tatooine_mpi_initialize_parameters and "
        "tatooine_mpi_initialize_variables");
  }
  log("Initializing tracking");

  _base_pmused = pm_used();
  _base_vmused = vm_used();

  // create output directory
  std::filesystem::create_directories(_output_dir_name);

  // TODO: max num particles per processor
  // central ids are 2*_num_ghost_particles + 1 apart because we need space
  // for the old positions of ghost particles after a reset

  if (*restart == 1) {
    // Append to log files
    _timings_file.open(_timings_fname, std::ios::app);
    _memory_file.open(_memory_fname, std::ios::app);
    _split_merge_file.open(_split_merge_fname, std::ios::app);
  } else {
    // Clear contents of log files
    _timings_file.open(_timings_fname, std::ios::trunc);
    _memory_file.open(_memory_fname, std::ios::trunc);
    _split_merge_file.open(_split_merge_fname, std::ios::trunc);
    // Seed particles on iso surface
  }
  _last_end_time        = std::chrono::system_clock::now();
  _tracking_initialized = true;
}
//------------------------------------------------------------------------------
auto tatooine_mpi_update_variables(double* mixture_fraction,
                                   double* prev_mixture_fraction,
                                   double* mixture_fraction_gradient,
                                   double* flow_velocity, double* heat_release)
    -> void {
  if (!_tracking_initialized) {
    throw std::logic_error(
        "tatooine_mpi_update_variables can only be called if "
        "initialization is complete");
  }
  log("Updating variables");
  // Here goes updating variables
}
//------------------------------------------------------------------------------
auto tatooine_mpi_update(int* iteration, double* time) -> void {
  if (!_tracking_initialized) {
    throw std::logic_error(
        "tatooine_mpi_update_tracking can only be called if "
        "initialization is complete");
  }
  _prev_time = _time;
  _time      = *time;
  _iteration = *iteration;
  log("Updating tracking for iteration " + std::to_string(_iteration));

  auto ct = std::chrono::system_clock::now();
  if (_mpi_comm->rank() == 0) {
    auto sim_time = ct - _last_end_time;
    _timings_file
        << _iteration << '\t'
        << std::chrono::duration_cast<std::chrono::milliseconds>(sim_time).count()
        << '\n';
  }

  // Here goes update

  log("Tatooine MPI update step finished");
  //log_mem_usage();
  _last_end_time = std::chrono::system_clock::now();
}
//==============================================================================
}  // namespace tatooine::mpi::interfaces::test
//==============================================================================
