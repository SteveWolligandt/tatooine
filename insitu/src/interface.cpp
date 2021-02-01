#include <tatooine/insitu/interface.h>
#include <tatooine/isosurface.h>
//==============================================================================
namespace tatooine::insitu {
//==============================================================================
std::filesystem::path interface::m_output_path = "tatooine_insitu_interface";
std::filesystem::path interface::m_isosurface_output_path =
    interface::m_output_path / "isosurfaces";
//==============================================================================
auto interface::initialize_velocity_x(double const* vel_x) -> void {
  init_par_and_var_check();
  m_velocity_x = &m_worker_halo_grid.create_vertex_property<scalar_arr_t>(
      "velocity_x", vel_x, m_worker_halo_grid.size(0),
      m_worker_halo_grid.size(1), m_worker_halo_grid.size(2));
}
//----------------------------------------------------------------------------
auto interface::initialize_velocity_y(double const* vel_y) -> void {
  init_par_and_var_check();
  m_velocity_y = &m_worker_halo_grid.create_vertex_property<scalar_arr_t>(
      "velocity_y", vel_y, m_worker_halo_grid.size(0),
      m_worker_halo_grid.size(1), m_worker_halo_grid.size(2));
}
//----------------------------------------------------------------------------
auto interface::initialize_velocity_z(double const* vel_z) -> void {
  init_par_and_var_check();
  m_velocity_z = &m_worker_halo_grid.create_vertex_property<scalar_arr_t>(
      "velocity_z", vel_z, m_worker_halo_grid.size(0),
      m_worker_halo_grid.size(1), m_worker_halo_grid.size(2));
}
//----------------------------------------------------------------------------
auto interface::initialize_parameters(double const time, double const prev_time,
                                      int const iteration) -> void {
  init_par_and_var_check();

  m_time      = time;
  m_prev_time = prev_time;
  m_iteration = iteration;
}
//----------------------------------------------------------------------------
auto interface::initialize(bool const restart) -> void {
  initialize_memory_file(restart, m_memory_fname);
  // create output directory
  std::filesystem::create_directories(m_output_path);
  std::filesystem::create_directories(m_isosurface_output_path);

  if (restart == 1) {
    // Append to log files
    m_timings_file.open(std::filesystem::path{m_timings_fname}, std::ios::app);
  } else {
    // Clear contents of log files
    m_timings_file.open(std::filesystem::path{m_timings_fname},
                        std::ios::trunc);
    // Seed particles on iso surface
  }

  m_velocity_field = std::make_unique<velocity_field>(
      m_worker_halo_grid, *m_velocity_x, *m_velocity_y, *m_velocity_z);
  m_phase = phase::initialized;
}
//----------------------------------------------------------------------------
auto interface::update_velocity_x(double const* var) -> void {
  update_var_check();
  m_velocity_x->change_data(var);
}
//----------------------------------------------------------------------------
auto interface::update_velocity_y(double const* var) -> void {
  update_var_check();
  m_velocity_y->change_data(var);
}
//----------------------------------------------------------------------------
auto interface::update_velocity_z(double const* var) -> void {
  update_var_check();
  m_velocity_z->change_data(var);
}
//----------------------------------------------------------------------------
auto interface::update(int const iteration, double const time) -> void {
  m_prev_time = m_time;
  m_time      = time;
  m_iteration = iteration;

  auto ct = std::chrono::system_clock::now();
  if (m_mpi_communicator->rank() == 0) {
    auto sim_time = ct - m_last_end_time;
    m_timings_file << m_iteration << '\t'
                   << std::chrono::duration_cast<std::chrono::milliseconds>(
                          sim_time)
                          .count()
                   << "ms\n";
  }

  extract_isosurfaces();
  m_last_end_time = std::chrono::system_clock::now();
}
//------------------------------------------------------------------------------
auto interface::extract_isosurfaces() -> void {
  extract_isosurfaces_velocity_magnitude();
}
//------------------------------------------------------------------------------
auto interface::extract_isosurfaces_velocity_magnitude() -> void {
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
  for (auto const iso : std::array{0, 1, 10, 20, 30, 40}) {
    isosurface(
        [&](auto const ix, auto const iy, auto const iz, auto const& /*pos*/) {
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
}
//==============================================================================
}  // namespace tatooine::insitu
//==============================================================================
