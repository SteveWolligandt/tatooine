#include <tatooine/dino_interface/base_interface.h>
#include <tatooine/dino_interface/interface.h>

#include <boost/multi_array.hpp>
#include <tatooine/multidim_array.h>
//==============================================================================
namespace tatooine::dino_interface {
//==============================================================================
struct interface : base_interface<test> {
  using this_t   = test;
  using parent_t = base_interface<this_t>;

  static constexpr std::string_view m_timings_fname     = "tatooine_dino_interface_timings.txt";
  static constexpr std::string_view m_memory_fname      = "tatooine_dino_interface_memory.txt";
  static constexpr std::string_view m_split_merge_fname = "tatooine_dino_interface_splitmerge.txt";

  static constexpr std::string_view m_output_dir_name = "tatooine_dino_interface_output";

  double   m_time      = 0;
  double   m_prev_time = 0;
  uint64_t m_iteration = 0;

  bool m_variables_initialized  = false;
  bool m_parameters_initialized = false;
  bool m_initialized            = false;

  std::ofstream m_timings_file;

  //==============================================================================
  // Interface Functions
  //==============================================================================
  auto initialize_variable(char const* /*name*/, int const num_components,
                           double const* var) -> void {
    if (m_variables_initialized) {
      return;
    }
    if (!m_grid_initialized) {
      throw std::logic_error(
          "initialize_variables must be called "
          "after initialize_grid");
    }
    log("Initializing variables");

    if (num_components == 1) {
      //using arr_t   = boost::multi_array<double, 3>;
      //arr_t transformed_data{
      //    boost::extents[m_worker_grid.size(0)][m_worker_grid.size(1)]
      //                  [m_worker_grid.size(2)]};
      //
      //size_t idx = 0;
      //for (size_t i = 0; i < m_worker_grid.size(0); ++i) {
      //  for (size_t j = 0; j < m_worker_grid.size(1); ++j) {
      //    for (size_t k = 0; k < m_worker_grid.size(2); ++k) {
      //      transformed_data[i][j][k] = var[idx++];
      //    }
      //  }
      //}
      //
      //for (size_t i = 0; i < 12; ++i) {
      //  if (m_mpi_communicator->rank() == 0) {
      //    std::cerr << transformed_data.data()[i] << ", ";
      //  }
      //  std::cerr << "...\n";
      //}
    } else {
      using arr_t   = dynamic_multidim_array<vec3>;

      arr_t transformed_data{m_worker_grid.size(0),
                             m_worker_grid.size(1),
                             m_worker_grid.size(2)};

      size_t idx = 0;
      for (size_t k = 0; k < m_worker_grid.size(2); ++k) {
        for (size_t j = 0; j < m_worker_grid.size(1); ++j) {
          for (size_t i = 0; i < m_worker_grid.size(0); ++i) {
            transformed_data(i, j, k) =
                vec3{var[idx], var[idx + 1], var[idx + 2]};
            idx += 3;
          }
        }
      }

      if (m_mpi_communicator->rank() == 0) {
        std::cerr << "from feeder: ";
        for (size_t i = 0; i < 12; ++i) {
          std::cerr << var[i] << ", ";
        }
        std::cerr << "...\n"
                  << "interface: ";
        for (size_t i = 0; i < 4; ++i) {
          std::cerr << transformed_data.data()[i].x() << ", ";
          std::cerr << transformed_data.data()[i].y() << ", ";
          std::cerr << transformed_data.data()[i].z() << ", ";
        }
        std::cerr << "...\n";
      }
    }
    m_variables_initialized = true;
  }
  //------------------------------------------------------------------------------
  auto initialize_parameters(double const time, double const prev_time,
                             int const iteration) -> void {
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
  auto initialize(bool const restart) -> void {
    initialize_memory_file(restart, m_memory_fname);
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
  auto update_variable(char const* /*name*/, int const /*num_components*/,
                       double const* /*var*/) -> void {
    if (!m_initialized) {
      throw std::logic_error(
          "update_variable can only be called if "
          "initialization is complete");
    }
    log("Updating variables");
    // Here goes updating variables
  }
  //------------------------------------------------------------------------------
  auto update(int const iteration, double const time) -> void {
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
    // log_mem_usage(m_iteration);
    m_last_end_time = std::chrono::system_clock::now();
  }
};
//==============================================================================
}  // namespace tatooine::dino_interface::interfaces
//==============================================================================

//==============================================================================
// Interface Functions
//==============================================================================
auto tatooine_dino_interface_initialize_communicator(MPI_Fint* communicator)
    -> void {
  tatooine::dino_interface::interfaces::test::get().initialize_communicator(
      *communicator);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize_grid(
    int const* global_grid_size_x, int const* global_grid_size_y,
    int const* global_grid_size_z, int const* local_starting_index_x,
    int const* local_starting_index_y, int const* local_starting_index_z,
    int const* local_grid_size_x, int const* local_grid_size_y,
    int const* local_grid_size_z, double const* domain_size_x,
    double const* domain_size_y, double const* domain_size_z,
    int const* is_periodic_x, int const* is_periodic_y,
    int const* is_periodic_z, int const* halo_level) -> void {
  tatooine::dino_interface::interfaces::test::get().initialize_grid(
      *global_grid_size_x, *global_grid_size_y, *global_grid_size_z,
      *local_starting_index_x, *local_starting_index_y, *local_starting_index_z,
      *local_grid_size_x, *local_grid_size_y, *local_grid_size_z,
      *domain_size_x, *domain_size_y, *domain_size_z, *is_periodic_x,
      *is_periodic_y, *is_periodic_z, *halo_level);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize_variable(char const*   name,
                                                 int const*    num_components,
                                                 double const* var) -> void {
  tatooine::dino_interface::interfaces::test::get().initialize_variable(
      name, *num_components, var);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize_parameters(double const* time,
                                                   double const* prev_time,
                                                   int const*    iteration)
    -> void {
  tatooine::dino_interface::interfaces::test::get().initialize_parameters(
      *time, *prev_time, *iteration);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize(int const* restart) -> void {
  tatooine::dino_interface::interfaces::test::get().initialize(*restart);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_update_variable(char const*   name,
                                             int const*    num_components,
                                             double const* var) -> void {
  tatooine::dino_interface::interfaces::test::get().update_variable(
      name, *num_components, var);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_update(int const* iteration, double const* time)
    -> void {
  tatooine::dino_interface::interfaces::test::get().update(*iteration, *time);
}
