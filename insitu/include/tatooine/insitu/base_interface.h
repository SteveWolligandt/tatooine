#ifndef TATOOINE_INSITU_BASE_INTERFACE_H
#define TATOOINE_INSITU_BASE_INTERFACE_H
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#include <tatooine/mpi/cartesian_neighbors.h>
#include <tatooine/netcdf.h>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <tatooine/filesystem.h>
#include <fstream>
#include <iomanip>
#include <iostream>
//==============================================================================
namespace tatooine::insitu {
//==============================================================================
template <typename InterfaceImplementation>
struct base_interface {
  static constexpr std::string_view reset       = "\033[0m";
  static constexpr std::string_view bold        = "\033[1m";
  static constexpr std::string_view black       = "\033[30m";
  static constexpr std::string_view red         = "\033[31m";
  static constexpr std::string_view green       = "\033[32m";
  static constexpr std::string_view yellow      = "\033[33m";
  static constexpr std::string_view blue        = "\033[34m";
  static constexpr std::string_view magenta     = "\033[35m";
  static constexpr std::string_view cyan        = "\033[36m";
  static constexpr std::string_view white       = "\033[37m";
  static constexpr std::string_view boldblack   = "\033[1m\033[30m";
  static constexpr std::string_view boldred     = "\033[1m\033[31m";
  static constexpr std::string_view boldgreen   = "\033[1m\033[32m";
  static constexpr std::string_view boldyellow  = "\033[1m\033[33m";
  static constexpr std::string_view boldblue    = "\033[1m\033[34m";
  static constexpr std::string_view boldmagenta = "\033[1m\033[35m";
  static constexpr std::string_view boldcyan    = "\033[1m\033[36m";
  static constexpr std::string_view boldwhite   = "\033[1m\033[37m";
  //============================================================================
  // Helper Functions
  //============================================================================
  static auto parse_line(char* line) {
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
  static auto vm_used() {  // Note: this value is in KB!
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
  static auto pm_used() {  // Note: this value is in KB!
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
  enum class phase : unsigned short {
    pre_start,
    initialized_communicator,
    initialized_grid,
    initializing_parameters_and_variables,
    initialized,
    preparing_update,
    updating
  };
  //============================================================================
  // MEMBERS
  //============================================================================
  std::unique_ptr<boost::mpi::cartesian_communicator> m_mpi_communicator;
  long                                                m_base_vmused = 0;
  long                                                m_base_pmused = 0;
  std::ofstream                                       m_memory_file;
  std::chrono::time_point<std::chrono::system_clock>  m_last_end_time;
  uniform_grid<double, 3>                             m_global_grid;
  uniform_grid<double, 3>                             m_worker_grid;
  uniform_grid<double, 3>                             m_worker_halo_grid;
  int                                                 m_halo_level = 0;
  double                                              m_time       = 0;
  double                                              m_prev_time  = 0;
  uint64_t                                            m_iteration  = 0;
  bool m_is_periodic_x = false, m_is_periodic_y = false,
       m_is_periodic_z = false;
  phase m_phase        = phase::pre_start;

  //============================================================================
  // METHODS
  //============================================================================
  auto initialize_memory_file(bool const                   restart,
                              filesystem::path const& filepath) -> void {
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
    if (m_phase >= phase::initialized_communicator) {
      return;
    }
    // Communicator should be the one describing the cartesian grid
    // of processors
    // Convert communicator to C
    m_mpi_communicator = std::unique_ptr<boost::mpi::cartesian_communicator>{
        new boost::mpi::cartesian_communicator{MPI_Comm_f2c(communicator),
                                               boost::mpi::comm_attach}};
    m_phase = phase::initialized_communicator;
  }
  //----------------------------------------------------------------------------
  void log(const std::string& message) {
    if (m_mpi_communicator->rank() == 0) {
      std::cout << bold << "[tatooine insitu interface]" << reset << "\n  "
                << message << '\n';
    }
  }
  //----------------------------------------------------------------------------
  void log_all(const std::string& message) {
    std::cout << bold << "[tatooine insitu interface#"
              << m_mpi_communicator->rank() << "]" << reset << "\n  " << message
              << '\n';
  }
  //----------------------------------------------------------------------------
  void log_mem_usage(int iteration) {
    // auto virtualMemUsedProcess = vm_used();
    auto mem_used_process     = pm_used();
    auto mem_overhead_process = mem_used_process - m_base_pmused;

    // Gather all memory usage at master node
    // auto vmused_all = std::vector<long>{};
    // boost::mpi::gather(*m_mpi_communicator, virtualMemUsedProcess,
    // vmused_all, 0L);

    auto pmused_all = std::vector<long>{};
    boost::mpi::gather(*m_mpi_communicator, mem_used_process, pmused_all, 0L);

    auto overhead_all = std::vector<long>{};
    boost::mpi::gather(*m_mpi_communicator, mem_overhead_process, overhead_all,
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

      // std::cout << "Virtual Memory: \n";
      // std::cout << "======================================\n";
      // std::cout << "Maximum per node: " << *(mima_vm.second) / 1024 << "Mb"
      //          << '\n';
      // std::cout << "Minimum per node: " << *(mima_vm.first) / 1024 << "Mb\n";
      // std::cout << "Average per node: " << vm_avg / 1024 << "Mb\n";
      // std::cout << "Total: " << vm_total / 1024 << "Mb\n";
      // std::cout << "Maximum baseline per node: " << *(mima_basevm.second) /
      // 1024
      //          << "Mb\n";
      // std::cout << "Minimum
      //    baseline per node : " << *(mima_basevm.first)/1024 << " Mb\n ";
      //                        std::cout
      //          << "Average baseline per node: "
      //          << basevm_avg / 1024
      //          << "Mb"
      //          << '\n';
      // std::cout << "Total baseline: " << basevm_total / 1024 << "Mb\n";
      // std::cout << '\n';

      std::cout << "Physical Memory: \n";
      std::cout << "======================================\n";
      std::cout << "Maximum per node: " << *(mima_pm.second) / 1024 << "Mb"
                << '\n';
      std::cout << "Minimum per node: " << *(mima_pm.first) / 1024 << "Mb"
                << '\n';
      std::cout << "Average per node: " << pm_avg / 1024 << "Mb\n";
      std::cout << "Total: " << pm_total / 1024 << "Mb\n";
      std::cout << "Maximum baseline per node: " << *(mima_basepm.second) / 1024
                << "Mb\n";
      std::cout << "Minimum baseline per node: " << *(mima_basepm.first) / 1024
                << "Mb\n";
      std::cout << "Average baseline per node: " << basepm_avg / 1024 << "Mb"
                << '\n';
      std::cout << "Total baseline: " << basepm_total / 1024 << "Mb\n";
      std::cout << "Maximum overhead per node: "
                << *(mima_overhead.second) / 1024 << "Mb\n";
      std::cout << "Minimum overhead per node: "
                << *(mima_overhead.first) / 1024 << "Mb\n";
      std::cout << "Average overhead per node: " << overhead_avg / 1024 << "Mb"
                << '\n';
      std::cout << "Total overhead: " << overhead_total / 1024 << "Mb" << '\n';
      std::cout << '\n';

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
  //==============================================================================
  auto init_par_and_var_check() -> void {
    if (m_phase < phase::initialized_grid) {
      throw std::logic_error(
          "[tatooine insitu interface]\n  "
          "Parameters and variables can first be initialized after "
          "initialize_grid");
    }
    m_phase = phase::initializing_parameters_and_variables;
  }
  //------------------------------------------------------------------------------
  auto update_var_check() -> void {
    if (m_phase < phase::initialized) {
      throw std::logic_error(
          "[tatooine insitu interface]\n  "
          "variables can first be update after initialize");
    }
    m_phase = phase::preparing_update;
  }
  //----------------------------------------------------------------------------
  template <typename T>
  auto mpi_gather(T const& in, int const root) const {
    std::vector<T> out;
    boost::mpi::gather(*m_mpi_communicator, in, out, root);
    return out;
  }
  //----------------------------------------------------------------------------
  template <typename T, typename ReceiveHandler>
  auto mpi_all_gather(std::vector<T> const& outgoing,
                      ReceiveHandler&&      receive_handler) const {
    std::vector<std::vector<T>> received_data;
    boost::mpi::all_gather(*m_mpi_communicator, outgoing, received_data);

    for (auto const& rec : received_data) {
      boost::for_each(rec, receive_handler);
    }
  }
  /// \brief Communicate a number of elements with all neighbor processes
  /// \details Sends a number of \p outgoing elements to all neighbors in the
  ///      given \p communicator. Receives a number of elements of the same type
  ///      from all neighbors. Calls the \p receive_handler for each received
  ///      element.
  ///
  /// \param outgoing Collection of elements to send to all neighbors
  /// \param comm Communicator for the communication
  /// \param receive_handler Functor that is called with each received element
  ///
  template <typename T, typename ReceiveHandler>
  auto mpi_gather_neighbors(std::vector<T> const& outgoing,
                            ReceiveHandler&&      receive_handler) -> void {
    namespace mpi = boost::mpi;
    auto sendreqs = std::vector<mpi::request>{};
    auto recvreqs = std::map<int, mpi::request>{};
    auto incoming = std::map<int, std::vector<T>>{};

    for (auto const& [rank, coords] : tatooine::mpi::cartesian_neighbors(
             m_mpi_communicator->coordinates(m_mpi_communicator->rank()),
             *m_mpi_communicator)) {
      incoming[rank] = std::vector<T>{};
      recvreqs[rank] = m_mpi_communicator->irecv(rank, rank, incoming[rank]);
      sendreqs.push_back(m_mpi_communicator->isend(
          rank, m_mpi_communicator->rank(), outgoing));
    }

    // wait for and handle receive requests
    while (!recvreqs.empty()) {
      for (auto& [key, req] : recvreqs) {
        if (req.test()) {
          boost::for_each(incoming[key], receive_handler);
          recvreqs.erase(key);
          break;
        }
      }
    }

    // wait for send requests to finish
    mpi::wait_all(begin(sendreqs), end(sendreqs));
  }
};
//==============================================================================
}  // namespace tatooine::insitu
//==============================================================================
#endif
