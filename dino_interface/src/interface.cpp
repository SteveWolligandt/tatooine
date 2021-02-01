#include <tatooine/dino_interface/base_interface.h>
#include <tatooine/dino_interface/interface.h>
#include <tatooine/isosurface.h>

#include <boost/multi_array.hpp>
#include <tatooine/multidim_array.h>
//==============================================================================
namespace tatooine::dino_interface {
//==============================================================================
struct interface : base_interface<interface> {
  using this_t   = interface;
  using parent_t = base_interface<this_t>;

  static constexpr std::string_view m_timings_fname =
      "tatooine_dino_interface_timings.txt";
  static constexpr std::string_view m_memory_fname =
      "tatooine_dino_interface_memory.txt";
  static constexpr std::string_view m_split_merge_fname =
      "tatooine_dino_interface_splitmerge.txt";
  static std::filesystem::path m_output_path;
  static std::filesystem::path m_isosurface_output_path;

  double   m_time      = 0;
  double   m_prev_time = 0;
  uint64_t m_iteration = 0;

  bool m_parameters_initialized = false;
  bool m_initialized            = false;

  using scalar_arr_t = non_owning_multidim_array<double, x_fastest>;
  using grid_prop_t =
      uniform_grid<double, 3>::typed_property_impl_t<scalar_arr_t>;

  grid_prop_t *m_velocity_x, *m_velocity_y, *m_velocity_z;

  std::ofstream m_timings_file;

  struct velocity_field : vectorfield<velocity_field, double, 3> {
    using this_t   = velocity_field;
    using parent_t = vectorfield<this_t, double, 3>;
    using parent_t::pos_t;
    using parent_t::real_t;
    using parent_t::tensor_t;

    uniform_grid<double, 3> const& m_worker_halo_grid;
    grid_prop_t const&             m_x;
    grid_prop_t const&             m_y;
    grid_prop_t const&             m_z;

    velocity_field(uniform_grid<double, 3> const& worker_halo_grid,
                   grid_prop_t const& x,
                   grid_prop_t const& y,
                   grid_prop_t const& z)
        : m_worker_halo_grid{worker_halo_grid}, m_x{x}, m_y{y}, m_z{z} {}

    auto evaluate(pos_t const& x, real_t const /*t*/) const
        -> tensor_t override {
      return {m_x.sampler<interpolation::linear>()(x),
              m_y.sampler<interpolation::linear>()(x),
              m_z.sampler<interpolation::linear>()(x)};
    }
    auto in_domain(pos_t const& x, real_t const /*t*/) const -> bool override {
      return m_worker_halo_grid.in_domain(x(0), x(1), x(2));
    }
  };

  std::unique_ptr<velocity_field> m_velocity_field;

  //============================================================================
  // Interface Functions
  //============================================================================
  auto initialize_velocity_x(double const* vel_x) -> void {
    if (!m_grid_initialized) {
      throw std::logic_error(
          "initialize_variables must be called "
          "after initialize_grid");
    }

    m_velocity_x = &m_worker_halo_grid.create_vertex_property<scalar_arr_t>(
        "velocity_x", vel_x, m_worker_halo_grid.size(0), m_worker_halo_grid.size(1),
        m_worker_halo_grid.size(2));
  }
  //----------------------------------------------------------------------------
  auto initialize_velocity_y(double const* vel_y) -> void {
    if (!m_grid_initialized) {
      throw std::logic_error(
          "initialize_variables must be called "
          "after initialize_grid");
    }
    m_velocity_y = &m_worker_halo_grid.create_vertex_property<scalar_arr_t>(
        "velocity_y", vel_y, m_worker_halo_grid.size(0), m_worker_halo_grid.size(1),
        m_worker_halo_grid.size(2));
  }
  //----------------------------------------------------------------------------
  auto initialize_velocity_z(double const* vel_z) -> void {
    if (!m_grid_initialized) {
      throw std::logic_error(
          "initialize_variables must be called "
          "after initialize_grid");
    }

    m_velocity_z = &m_worker_halo_grid.create_vertex_property<scalar_arr_t>(
        "velocity_z", vel_z, m_worker_halo_grid.size(0), m_worker_halo_grid.size(1),
        m_worker_halo_grid.size(2));
  }
  //----------------------------------------------------------------------------
  auto initialize_parameters(double const time, double const prev_time,
                             int const iteration) -> void {
    if (m_parameters_initialized) { return; }
    if (!m_grid_initialized) {
      throw std::logic_error(
          "initialize_parameters must be called "
          "after initialize_grid");
    }
    log("[TATOOINE] Initializing parameters");
    m_time      = time;
    m_prev_time = prev_time;
    m_iteration = iteration;

    m_parameters_initialized = true;
  }
  //----------------------------------------------------------------------------
  auto initialize(bool const restart) -> void {
    initialize_memory_file(restart, m_memory_fname);
    if (m_initialized) {
      return;
    }
    if (!m_parameters_initialized) {
      throw std::logic_error(
          "initialize must be called "
          "after initialize_parameters and "
          "initialize_variables");
    }
    //log("Initializing");

    // create output directory
    std::filesystem::create_directories(m_output_path);
    std::filesystem::create_directories(m_isosurface_output_path);

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

    m_velocity_field = std::make_unique<velocity_field>(
        m_worker_halo_grid, *m_velocity_x, *m_velocity_y, *m_velocity_z);
    m_initialized = true;
  }
  //----------------------------------------------------------------------------
  auto update_velocity_x(double const* var) -> void {
    if (!m_initialized) {
      throw std::logic_error(
          "update_variable can only be called if "
          "initialization is complete");
    }
    m_velocity_x->change_data(var);
  }
  //----------------------------------------------------------------------------
  auto update_velocity_y(double const* var) -> void {
    if (!m_initialized) {
      throw std::logic_error(
          "update_variable can only be called if "
          "initialization is complete");
    }
    m_velocity_y->change_data(var);
  }
  //----------------------------------------------------------------------------
  auto update_velocity_z(double const* var) -> void {
    if (!m_initialized) {
      throw std::logic_error(
          "update_variable can only be called if "
          "initialization is complete");
    }
    m_velocity_z->change_data(var);
  }
  //----------------------------------------------------------------------------
  auto update(int const iteration, double const time) -> void {
    if (!m_initialized) {
      throw std::logic_error(
          "update can only be called if "
          "initialization is complete");
    }
    m_prev_time = m_time;
    m_time      = time;
    m_iteration = iteration;
    //log("Updating for iteration " + std::to_string(m_iteration));

    auto ct = std::chrono::system_clock::now();
    //if (m_mpi_communicator->rank() == 0) {
    //  auto sim_time = ct - m_last_end_time;
    //  m_timings_file << m_iteration << '\t'
    //                 << std::chrono::duration_cast<std::chrono::milliseconds>(
    //                        sim_time)
    //                        .count()
    //                 << '\n';
    //}

    auto isogrid = m_worker_grid;
    if (std::abs(isogrid.dimension<0>().back() -
                 m_global_grid.dimension<0>().back()) > 1e-6) {
      isogrid.dimension<0>().push_back();
    }
    if (std::abs(isogrid.dimension<1>().back() -
                 m_global_grid.dimension<1>().back()) > 1e-6) {
      isogrid.dimension<1>().push_back();
    }
    if (std::abs(isogrid.dimension<2>().back() -
                 m_global_grid.dimension<2>().back()) > 1e-6) {
      isogrid.dimension<2>().push_back();
    }
    for (auto const iso : std::array{10, 20, 30, 40}) {
      isosurface(
          [&](auto const ix, auto const iy, auto const iz,
              auto const& /*pos*/) {
            auto const velx = m_velocity_x->at(ix, iy, iz);
            auto const vely = m_velocity_y->at(ix, iy, iz);
            auto const velz = m_velocity_z->at(ix, iy, iz);
            return std::sqrt(velx * velx + vely * vely + velz * velz);
          },
          m_worker_halo_grid, iso)
          .write_vtk(m_isosurface_output_path /
                     std::filesystem::path{
                         "vel_mag_" + std::to_string(iso) + "_rank_" +
                         std::to_string(m_mpi_communicator->rank()) + "_time_" +
                         std::to_string(m_iteration) + ".vtk"});
    }

    //log("Tatooine update step finished");
    m_last_end_time = std::chrono::system_clock::now();
  }
};
std::filesystem::path interface::m_output_path = "tatooine_insitu";
std::filesystem::path interface::m_isosurface_output_path =
    interface::m_output_path / "isosurfaces";
//==============================================================================
}  // namespace tatooine::dino_interface
//==============================================================================

//==============================================================================
// Interface Functions
//==============================================================================
auto tatooine_dino_interface_initialize_communicator(MPI_Fint* communicator)
    -> void {
  tatooine::dino_interface::interface::get().initialize_communicator(
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
  tatooine::dino_interface::interface::get().initialize_grid(
      *global_grid_size_x, *global_grid_size_y, *global_grid_size_z,
      *local_starting_index_x, *local_starting_index_y, *local_starting_index_z,
      *local_grid_size_x, *local_grid_size_y, *local_grid_size_z,
      *domain_size_x, *domain_size_y, *domain_size_z, *is_periodic_x,
      *is_periodic_y, *is_periodic_z, *halo_level);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize_velocity_x(double const* vel_x) -> void {
  tatooine::dino_interface::interface::get().initialize_velocity_x(vel_x);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize_velocity_y(double const* vel_y) -> void {
  tatooine::dino_interface::interface::get().initialize_velocity_y(vel_y);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize_velocity_z(double const* vel_z) -> void {
  tatooine::dino_interface::interface::get().initialize_velocity_z(vel_z);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize_parameters(double const* time,
                                                   double const* prev_time,
                                                   int const*    iteration)
    -> void {
  tatooine::dino_interface::interface::get().initialize_parameters(
      *time, *prev_time, *iteration);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_initialize(int const* restart) -> void {
  tatooine::dino_interface::interface::get().initialize(*restart);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_update_velocity_x(double const* vel_x) -> void {
  tatooine::dino_interface::interface::get().update_velocity_x(vel_x);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_update_velocity_y(double const* vel_y) -> void {
  tatooine::dino_interface::interface::get().update_velocity_y(vel_y);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_update_velocity_z(double const* vel_z) -> void {
  tatooine::dino_interface::interface::get().update_velocity_z(vel_z);
}
//------------------------------------------------------------------------------
auto tatooine_dino_interface_update(int const* iteration, double const* time)
    -> void {
  tatooine::dino_interface::interface::get().update(*iteration, *time);
}
