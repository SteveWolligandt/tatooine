#ifndef TATOOINE_DINO_INTERFACE_BASE_INTERFACE_H
#define TATOOINE_DINO_INTERFACE_BASE_INTERFACE_H
//==============================================================================
#include <tatooine/grid.h>

#include <boost/mpi.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
//==============================================================================
namespace tatooine::dino_interface {
//==============================================================================
template <typename InterfaceImplementation>
struct base_interface {
  //============================================================================
  // Helper Functions
  //============================================================================
  static long parse_line(char* line) {
    // This assumes that a digit will be found and the line ends in " Kb".
    auto        i = strlen(line);
    const char* p = line;
    while (*p < '0' || *p > '9')
      p++;
    line[i - 3] = '\0';
    i           = std::atol(p);
    return long(i);
  }
  //----------------------------------------------------------------------------
  static long vm_used() {  // Note: this value is in KB!
    FILE* file   = fopen("/proc/self/status", "r");
    long  result = -1;
    char  line[128];

    while (fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "VmSize:", 7) == 0) {
        result = parse_line(line);
        break;
      }
    }
    fclose(file);
    return result;
  }
  //----------------------------------------------------------------------------
  static long pm_used() {  // Note: this value is in KB!
    FILE* file   = fopen("/proc/self/status", "r");
    long  result = -1;
    char  line[128];

    while (fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "VmRSS:", 6) == 0) {
        result = parse_line(line);
        break;
      }
    }
    fclose(file);
    return result;
  }
  //============================================================================
  static auto get() -> auto& {
    static InterfaceImplementation impl;
    return impl;
  }
  //============================================================================
  // MEMBERS
  //============================================================================
  bool m_mpi_communicator_initialized = false;
  bool m_grid_initialized             = false;
  std::unique_ptr<boost::mpi::cartesian_communicator> m_mpi_communicator;
  long                                                m_base_vmused = 0;
  long                                                m_base_pmused = 0;
  std::ofstream                                       m_memory_file;
  std::chrono::time_point<std::chrono::system_clock>  m_last_end_time;
  uniform_grid<double, 3>                             m_global_grid;
  uniform_grid<double, 3>                             m_worker_grid;
  uniform_grid<double, 3>                             m_worker_halo_grid;
  int                                                 m_halo_level = 0;

  //============================================================================
  // METHODS
  //============================================================================
  auto initialize_memory_file(bool const                   restart,
                              std::filesystem::path const& filepath) -> void {
    m_base_pmused = pm_used();
    m_base_vmused = vm_used();

    if (restart) {
      // Clear contents of log files
      m_memory_file.open(filepath, std::ios::trunc);
      // Seed particles on iso surface
    } else {
      // Append to log files
      m_memory_file.open(filepath, std::ios::app);
    }
    m_base_pmused   = pm_used();
    m_base_vmused   = vm_used();
    m_last_end_time = std::chrono::system_clock::now();
  }
  //------------------------------------------------------------------------------
  auto initialize_communicator(MPI_Fint& communicator) -> void {
    if (m_mpi_communicator_initialized) {
      return;
    }
    // Communicator should be the one describing the cartesian grid
    // of processors
    // Convert communicator to C
    m_mpi_communicator = std::unique_ptr<boost::mpi::cartesian_communicator>{
        new boost::mpi::cartesian_communicator{MPI_Comm_f2c(communicator),
                                               boost::mpi::comm_attach}};
    log("Initializing MPI");
    m_mpi_communicator_initialized = true;
  }
  //------------------------------------------------------------------------------
  /// \brief  Initialize the dataset grid.
  ///
  /// \param  global_grid_size_x, global_grid_size_y, global_grid_size_z global
  /// grid dimensions \param  local_starting_index_x, local_starting_index_y,
  /// local_starting_index_z           starting indices of current
  ///                                 process
  /// \param  local_grid_size_x, local_grid_size_y, local_grid_size_z number of
  /// grid points of current process \param  domain_size_x, domain_size_y,
  /// domain_size_z              size of domain box \param  is_periodic_x,
  /// is_periodic_y, is_periodic_z     periodic boundary directions
  ///                                             (0 for no, 1 for yes)
  /// \param  halo_level              number of halo cell layers
  auto initialize_grid(
      int const global_grid_size_x, int const global_grid_size_y,
      int const global_grid_size_z, int const local_starting_index_x,
      int const local_starting_index_y, int const local_starting_index_z,
      int const local_grid_size_x, int const local_grid_size_y,
      int const local_grid_size_z, double const domain_size_x,
      double const domain_size_y, double const domain_size_z,
      int const /*is_periodic_x*/, int const /*is_periodic_y*/,
      int const /*is_periodic_z*/, int const halo_level) -> void {
    if (m_grid_initialized) {
      return;
    }
    if (!m_mpi_communicator_initialized) {
      throw std::logic_error(
          "initialize_grid must be called after "
          "initialize");
    }
    log("Initializing grid");

    assert(global_grid_size_x >= 0);
    assert(global_grid_size_y >= 0);
    assert(global_grid_size_z >= 0);
    assert(local_grid_size_x >= 0);
    assert(local_grid_size_y >= 0);
    assert(local_grid_size_z >= 0);
    assert(domain_size_x >= 0);
    assert(domain_size_y >= 0);
    assert(domain_size_z >= 0);
    assert(halo_level >= 0 && halo_level <= UINT8_MAX);

    //log_all("global_grid_size_x: " + std::to_string(global_grid_size_x));
    //log_all("global_grid_size_y: " + std::to_string(global_grid_size_y));
    //log_all("global_grid_size_z: " + std::to_string(global_grid_size_z));
    //log_all("local_starting_index_x: " +
    //        std::to_string(local_starting_index_x));
    //log_all("local_starting_index_y: " +
    //        std::to_string(local_starting_index_y));
    //log_all("local_starting_index_z: " +
    //        std::to_string(local_starting_index_z));
    //log_all("domain_size_x: " + std::to_string(domain_size_x));
    //log_all("domain_size_y: " + std::to_string(domain_size_y));
    //log_all("domain_size_z: " + std::to_string(domain_size_z));
    //log_all("halo_level: " + std::to_string(halo_level));

    if (halo_level < 4) {
      throw std::invalid_argument("halo_level must be at least 4. Given: " +
                                  std::to_string(halo_level));
    }

    m_global_grid.dimension<0>() = linspace<double>{
        0, domain_size_x, static_cast<size_t>(global_grid_size_x)};
    m_global_grid.dimension<1>() = linspace<double>{
        0, domain_size_y, static_cast<size_t>(global_grid_size_y)};
    m_global_grid.dimension<2>() = linspace<double>{
        0, domain_size_z, static_cast<size_t>(global_grid_size_z)};

    m_worker_grid.dimension<0>() = linspace{
        m_global_grid.dimension<0>()[local_starting_index_x],
        m_global_grid
            .dimension<0>()[local_starting_index_x + local_grid_size_x - 1],
        static_cast<size_t>(local_grid_size_x)};
    m_worker_grid.dimension<1>() = linspace{
        m_global_grid.dimension<1>()[local_starting_index_y],
        m_global_grid
            .dimension<1>()[local_starting_index_y + local_grid_size_y - 1],
        static_cast<size_t>(local_grid_size_y)};
    m_worker_grid.dimension<2>() = linspace{
        m_global_grid.dimension<2>()[local_starting_index_z],
        m_global_grid
            .dimension<2>()[local_starting_index_z + local_grid_size_z - 1],
        static_cast<size_t>(local_grid_size_z)};

    m_worker_halo_grid.dimension<0>() = linspace{
        m_global_grid.dimension<0>()[local_starting_index_x],
        m_global_grid
            .dimension<0>()[local_starting_index_x + local_grid_size_x - 1],
        static_cast<size_t>(local_grid_size_x)};

    // no pencil in x-direction
    if (local_grid_size_x < global_grid_size_x) {
      for (int i = 0; i < halo_level; ++i) {
        m_worker_halo_grid.dimension<0>().push_front();
      }
      for (int i = 0; i < halo_level; ++i) {
        m_worker_halo_grid.dimension<0>().push_back();
      }
    }
    m_worker_halo_grid.dimension<1>() = linspace{
        m_global_grid.dimension<1>()[local_starting_index_y],
        m_global_grid
            .dimension<1>()[local_starting_index_y + local_grid_size_y - 1],
        static_cast<size_t>(local_grid_size_y)};

    // no pencil in y-direction
    if (local_grid_size_y < global_grid_size_y) {
      for (int i = 0; i < halo_level; ++i) {
        m_worker_halo_grid.dimension<1>().push_front();
      }
      for (int i = 0; i < halo_level; ++i) {
        m_worker_halo_grid.dimension<1>().push_back();
      }
    }

    m_worker_halo_grid.dimension<2>() = linspace{
        m_global_grid.dimension<2>()[local_starting_index_z],
        m_global_grid
            .dimension<2>()[local_starting_index_z + local_grid_size_z - 1],
        static_cast<size_t>(local_grid_size_z)};
    // no pencil in z-direction
    if (local_grid_size_z < global_grid_size_z) {
      for (int i = 0; i < halo_level; ++i) {
        m_worker_halo_grid.dimension<2>().push_front();
      }
      for (int i = 0; i < halo_level; ++i) {
        m_worker_halo_grid.dimension<2>().push_back();
      }
    }

    if (m_mpi_communicator->rank() == 0) {
      std::cerr << "global grid:\n"
                << m_global_grid.dimension<0>() << '\n'
                << m_global_grid.dimension<1>() << '\n'
                << m_global_grid.dimension<2>() << '\n';
      std::cerr << "worker grid:\n"
                << m_worker_grid.dimension<0>() << '\n'
                << m_worker_grid.dimension<1>() << '\n'
                << m_worker_grid.dimension<2>() << '\n';
      std::cerr << "worker halo grid:\n"
                << m_worker_halo_grid.dimension<0>() << '\n'
                << m_worker_halo_grid.dimension<1>() << '\n'
                << m_worker_halo_grid.dimension<2>() << '\n';
    }
    m_halo_level       = halo_level;
    m_grid_initialized = true;
  }
  //----------------------------------------------------------------------------
  void log(const std::string& message) {
    if (m_mpi_communicator->rank() == 0) {
      std::cerr << message << '\n';
    }
  }
  //----------------------------------------------------------------------------
  void log_all(const std::string& message) {
    std::cerr << m_mpi_communicator->rank() << ": " << message << '\n';
  }
  //----------------------------------------------------------------------------
  void log_mem_usage(int iteration) {
    // auto virtualMemUsedProcess = vm_used();
    auto memUsedProcess     = pm_used();
    auto memOverheadProcess = memUsedProcess - m_base_pmused;

    // Gather all memory usage at master node
    // auto vmused_all = std::vector<long>{};
    // boost::mpi::gather(*m_mpi_communicator, virtualMemUsedProcess,
    // vmused_all, 0L);

    auto pmused_all = std::vector<long>{};
    boost::mpi::gather(*m_mpi_communicator, memUsedProcess, pmused_all, 0L);

    auto overhead_all = std::vector<long>{};
    boost::mpi::gather(*m_mpi_communicator, memOverheadProcess, overhead_all,
                       0L);

    // auto basevm_all = std::vector<long>{};
    // boost::mpi::gather(*m_mpi_communicator, _base_vmused, basevm_all, 0L);

    auto basepm_all = std::vector<long>{};
    boost::mpi::gather(*m_mpi_communicator, m_base_pmused, basepm_all, 0L);

    if (m_mpi_communicator->rank() == 0) {
      // auto mima_vm = std::minmax_element(begin(vmused_all),
      //                                    end(vmused_all));
      // auto vm_total = std::accumulate(begin(vmused_all),
      //                                 end(vmused_all), 0LL);
      // auto vm_avg = vm_total / m_mpi_communicator->size();

      auto mima_pm  = std::minmax_element(begin(pmused_all), end(pmused_all));
      auto pm_total = std::accumulate(begin(pmused_all), end(pmused_all), 0LL);
      auto pm_avg   = pm_total / m_mpi_communicator->size();

      auto mima_overhead =
          std::minmax_element(begin(overhead_all), end(overhead_all));
      auto overhead_total =
          std::accumulate(begin(overhead_all), end(overhead_all), 0LL);
      auto overhead_avg = overhead_total / m_mpi_communicator->size();

      // auto mima_basevm = std::minmax_element(begin(basevm_all),
      //                                        end(basevm_all));
      // auto basevm_total = std::accumulate(begin(basevm_all),
      //                                     end(basevm_all), 0LL);
      // auto basevm_avg = basevm_total / m_mpi_communicator->size();

      auto mima_basepm =
          std::minmax_element(begin(basepm_all), end(basepm_all));
      auto basepm_total =
          std::accumulate(begin(basepm_all), end(basepm_all), 0LL);
      auto basepm_avg = basepm_total / m_mpi_communicator->size();

      // auto numtr_avg   = numtr_total / m_mpi_communicator->size();

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
      std::cerr << "Maximum overhead per node: "
                << *(mima_overhead.second) / 1024 << "Mb\n";
      std::cerr << "Minimum overhead per node: "
                << *(mima_overhead.first) / 1024 << "Mb\n";
      std::cerr << "Average overhead per node: " << overhead_avg / 1024 << "Mb"
                << '\n';
      std::cerr << "Total overhead: " << overhead_total / 1024 << "Mb" << '\n';
      std::cerr << '\n';

      m_memory_file << iteration
                    << '\t'
                    //<< numtr_total << '\t'
                    //<< *(mima_numtr.second) << '\t' << numtr_avg << '\t'
                    << pm_total / 1024 << '\t' << *(mima_pm.second) / 1024
                    << '\t' << *(mima_pm.first) / 1024 << '\t' << pm_avg / 1024
                    << '\t' << overhead_total / 1024 << '\t'
                    << *(mima_overhead.second) / 1024 << '\t'
                    << *(mima_overhead.first) / 1024 << '\t'
                    << overhead_avg / 1024 << '\n';
    }
  }
  //============================================================================
  auto initialize_variable(char const* name, int num_components,
                           double const* var) -> void {
    static_cast<InterfaceImplementation*>(this)->initialize_variable(
        name, num_components, var);
  }
  //----------------------------------------------------------------------------
  auto update_variable(char const* name, int const num_components,
                       double const* var) -> void {
    static_cast<InterfaceImplementation*>(this)->update_variable(
        name, num_components, var);
  }
};
//==============================================================================
}  // namespace tatooine::dino_interface
//==============================================================================
#endif
