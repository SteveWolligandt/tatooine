#include <tatooine/insitu/interface.h>
#include <tatooine/isosurface.h>
#include <tatooine/vtk_legacy.h>

#include <boost/serialization/variant.hpp>
#include <cmath>
#include <fstream>
//==============================================================================
namespace tatooine::insitu {
//==============================================================================
std::filesystem::path interface::m_output_path = "tatooine_insitu_interface";
std::filesystem::path interface::m_isosurface_output_path =
    interface::m_output_path / "isosurfaces";
std::filesystem::path interface::m_tracers_output_path =
    interface::m_output_path / "tracers";
std::filesystem::path interface::m_tracers_tmp_path =
    interface::m_tracers_output_path / "tmp";
//==============================================================================
auto interface::initialize_grid(
    int const global_grid_size_x, int const global_grid_size_y,
    int const global_grid_size_z, int const local_starting_index_x,
    int const local_starting_index_y, int const local_starting_index_z,
    int const local_grid_size_x, int const local_grid_size_y,
    int const local_grid_size_z, double const domain_size_x,
    double const domain_size_y, double const domain_size_z,
    int const is_periodic_x, int const is_periodic_y, int const is_periodic_z,
    int const halo_level) -> void {
  if (m_phase < phase::initialized_communicator) {
    throw std::logic_error(
        "[tatooine insitu interface]\n  "
        "initialize_grid must be called after initialize_communicator");
  }
  // log("Initializing grid");

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

  // log_all(" global_grid_size_x: " + std::to_string(global_grid_size_x));
  // log_all(" global_grid_size_y: " + std::to_string(global_grid_size_y));
  // log_all(" global_grid_size_z: " + std::to_string(global_grid_size_z));
  // log_all(" local_starting_index_x: " +
  //        std::to_string(local_starting_index_x));
  // log_all(" local_starting_index_y: " +
  //        std::to_string(local_starting_index_y));
  // log_all(" local_starting_index_z: " +
  //        std::to_string(local_starting_index_z));
  // log_all(" domain_size_x: " + std::to_string(domain_size_x));
  // log_all(" domain_size_y: " + std::to_string(domain_size_y));
  // log_all(" domain_size_z: " + std::to_string(domain_size_z));
  // log_all(" halo_level: " + std::to_string(halo_level));

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

  m_halo_level    = halo_level;
  m_is_periodic_x = is_periodic_x;
  m_is_periodic_y = is_periodic_y;
  m_is_periodic_z = is_periodic_z;
  m_phase         = phase::initialized_grid;

  auto const bb = m_worker_grid.bounding_box();
  for (size_t i = 0; i < m_num_tracers; ++i) {
    auto const& [idx, pos] = m_tracers.emplace_back(
        m_mpi_communicator->rank() * m_num_tracers + i, bb.random_point());
    std::fstream fout{m_tracers_tmp_path / (std::to_string(idx) + ".bin"),
                      std::ios::binary | std::ios::out | std::ios::trunc};
    fout.write(reinterpret_cast<char const*>(pos.data_ptr()),
               sizeof(double) * 3);
  }
}
//----------------------------------------------------------------------------
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
  namespace fs = std::filesystem;
  initialize_memory_file(restart, m_memory_fname);
  // create output directory
  fs::create_directories(m_output_path);
  fs::create_directories(m_isosurface_output_path);
  fs::create_directories(m_tracers_output_path);
  fs::create_directories(m_tracers_tmp_path);

  if (restart == 1) {
    // Append to log files
    m_timings_file.open(fs::path{m_timings_fname}, std::ios::app);
  } else {
    // Clear contents of log files
    m_timings_file.open(fs::path{m_timings_fname}, std::ios::trunc);
    // Seed particles on iso surface
  }

  m_velocity_field = std::make_unique<velocity_field>(
      m_worker_halo_grid, *m_velocity_x, *m_velocity_y, *m_velocity_z);
  m_phase = phase::initialized;

  extract_isosurfaces();
  advect_tracers();

  for (auto const& [idx, pos] : m_tracers) {
    std::fstream fout{m_tracers_tmp_path / (std::to_string(idx) + ".bin"),
                      std::ios::binary | std::ios::out | std::ios::app};
    fout.write(reinterpret_cast<char const*>(pos.data_ptr()),
               sizeof(double) * 3);
  }
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
  advect_tracers();
  for (auto const& [idx, pos] : m_tracers) {
    std::fstream fout{m_tracers_tmp_path / (std::to_string(idx) + ".bin"),
                      std::ios::binary | std::ios::out | std::ios::app};
    fout.write(reinterpret_cast<char const*>(pos.data_ptr()),
               sizeof(double) * 3);
  }

  if (m_mpi_communicator->rank() == 0) {
    create_tracer_vtk();
  }

  m_last_end_time = std::chrono::system_clock::now();
}
//----------------------------------------------------------------------------
auto interface::advect_tracer(interface::pos_t& tracer_pos) -> bool {
  tracer_pos +=
      (m_time - m_prev_time) * m_velocity_field->evaluate(tracer_pos, m_time);

  if (tracer_pos.x() > m_global_grid.dimension<0>().back()) {
    if (m_is_periodic_x) {
      tracer_pos.x() =
          fmod(tracer_pos.x(), m_global_grid.dimension<0>().back());
    } else {
      return false;
    }
  }
  if (tracer_pos.x() < m_global_grid.dimension<0>().front()) {
    if (m_is_periodic_x) {
      while (tracer_pos.x() < m_global_grid.dimension<0>().front()) {
        tracer_pos.x() += m_global_grid.dimension<0>().back();
      }
    } else {
      return false;
    }
  }
  if (tracer_pos.y() > m_global_grid.dimension<1>().back()) {
    if (m_is_periodic_y) {
      tracer_pos.y() =
          fmod(tracer_pos.y(), m_global_grid.dimension<1>().back());
    } else {
      return false;
    }
  }
  if (tracer_pos.y() < m_global_grid.dimension<1>().front()) {
    if (m_is_periodic_y) {
      while (tracer_pos.y() < m_global_grid.dimension<1>().front()) {
        tracer_pos.y() += m_global_grid.dimension<1>().back();
      }
    } else {
      return false;
    }
  }
  if (tracer_pos.z() > m_global_grid.dimension<2>().back()) {
    if (m_is_periodic_z) {
      tracer_pos.z() =
          fmod(tracer_pos.z(), m_global_grid.dimension<2>().back());
    } else {
      return false;
    }
  }
  if (tracer_pos.z() < m_global_grid.dimension<2>().front()) {
    if (m_is_periodic_z) {
      while (tracer_pos.z() < m_global_grid.dimension<2>().front()) {
        tracer_pos.z() += m_global_grid.dimension<1>().back();
      }
    } else {
      return false;
    }
  }

  return true;
}
//----------------------------------------------------------------------------
auto interface::advect_tracers() -> void {
  auto advected_positions = std::move(m_tracers);
  m_tracers.reserve(size(advected_positions));
  for (auto& [idx, tracer_pos] : advected_positions) {
    if (advect_tracer(tracer_pos)) {
      m_tracers.emplace_back(idx, tracer_pos);
    }
  }
  std::vector<std::pair<size_t, pos_t>> in_working_area;

  auto bb = m_worker_grid.bounding_box();
  bb.min(0) -= m_worker_grid.dimension<0>().spacing() / 2;
  bb.max(0) += m_worker_grid.dimension<0>().spacing() / 2;
  bb.min(1) -= m_worker_grid.dimension<1>().spacing() / 2;
  bb.max(1) += m_worker_grid.dimension<1>().spacing() / 2;
  bb.min(2) -= m_worker_grid.dimension<2>().spacing() / 2;
  bb.max(2) += m_worker_grid.dimension<2>().spacing() / 2;
  size_t cnt = 0;
  mpi_all_gather_neighbors(m_tracers, [&](auto const& x) {
    ++cnt;
    if (bb.is_inside(x.second)) {
      in_working_area.push_back(x);
    }
  });
  log_all(std::to_string(cnt));
  m_tracers = std::move(in_working_area);
}
//------------------------------------------------------------------------------
auto interface::create_tracer_vtk() -> void {
  namespace fs = std::filesystem;
  vtk::legacy_file_writer tracer_collector{
      m_tracers_output_path / "tracers.vtk", vtk::dataset_type::polydata};

  size_t                                   total_num_points = 0;
  size_t                                   num_lines        = 0;
  size_t                                   lines_size       = 0;
  std::vector<std::pair<size_t, fs::path>> tracer_info;

  for (auto const& bin_tracer_dir_entry :
       fs::directory_iterator(m_tracers_tmp_path)) {
    auto const tmp_filesize   = fs::file_size(bin_tracer_dir_entry);
    auto const cur_num_points = tmp_filesize / (sizeof(double) * 3);
    if (cur_num_points > 1) {
      tracer_info.emplace_back(cur_num_points, bin_tracer_dir_entry.path());
      total_num_points += cur_num_points;
      lines_size += cur_num_points + 1;
      ++num_lines;
    }
  }

  // write
  tracer_collector.set_title("tatooine insitu tracer");
  tracer_collector.write_header();
  tracer_collector.close();
  std::fstream tracer_vtk_file{
      m_tracers_output_path / "tracers.vtk",
      std::ios::binary | std::ios::out | std::ios::app};

  // write points
  std::stringstream points_header_stream;
  points_header_stream << "\nPOINTS " << total_num_points << ' '
                       << type_to_str<double>() << '\n';
  std::copy(std::istreambuf_iterator<char>(points_header_stream),
            std::istreambuf_iterator<char>(),
            std::ostreambuf_iterator<char>(tracer_vtk_file));
  for (auto const& [cur_num_points, path] : tracer_info) {
    auto const          num_entries   = cur_num_points * 3;
    auto const          tmp_file_size = num_entries * sizeof(double);
    std::fstream        tracer_tmp_file{path, std::ios::binary | std::ios::in};
    std::vector<double> points(num_entries);
    tracer_tmp_file.read(reinterpret_cast<char*>(points.data()), tmp_file_size);
    swap_endianess(points);
    tracer_vtk_file.write(reinterpret_cast<char*>(points.data()),
                          tmp_file_size);
  }

  // write line indices
  std::stringstream lines_header_stream;
  lines_header_stream << "\nLINES " << num_lines << ' ' << lines_size << '\n';
  std::copy(std::istreambuf_iterator<char>(lines_header_stream),
            std::istreambuf_iterator<char>(),
            std::ostreambuf_iterator<char>(tracer_vtk_file));
  size_t cur_initial_index = 0;
  for (auto const& [cur_num_points, path] : tracer_info) {
    // write number of indices and indices of current line
    std::vector<int> cur_line(cur_num_points + 1);
    cur_line.front() = cur_num_points;
    std::iota(next(begin(cur_line)), end(cur_line), cur_initial_index);
    swap_endianess(cur_line);
    tracer_vtk_file.write(reinterpret_cast<char*>(cur_line.data()),
                          sizeof(int) * cur_line.size());
    cur_initial_index += cur_num_points;
  }
}
//------------------------------------------------------------------------------
auto interface::extract_isosurfaces() -> void {
  extract_isosurfaces_velocity_magnitude();
}
//------------------------------------------------------------------------------
auto interface::extract_isosurfaces_velocity_magnitude() -> void {
  namespace fs = std::filesystem;
  auto isogrid = m_worker_grid;
  // if (std::abs(isogrid.dimension<0>().back() -
  //             m_global_grid.dimension<0>().back()) > 1e-6) {
  //  isogrid.dimension<0>().push_back();
  //}
  // if (std::abs(isogrid.dimension<1>().back() -
  //             m_global_grid.dimension<1>().back()) > 1e-6) {
  //  isogrid.dimension<1>().push_back();
  //}
  // if (std::abs(isogrid.dimension<2>().back() -
  //             m_global_grid.dimension<2>().back()) > 1e-6) {
  //  isogrid.dimension<2>().push_back();
  //}
  for (auto const iso : std::array{0, 1}) {
    isosurface(
        [&](auto const ix, auto const iy, auto const iz, auto const& /*pos*/) {
          auto const velx = m_velocity_x->at(ix, iy, iz);
          auto const vely = m_velocity_y->at(ix, iy, iz);
          auto const velz = m_velocity_z->at(ix, iy, iz);
          return std::sqrt(velx * velx + vely * vely + velz * velz);
        },
        m_worker_halo_grid, iso)
        .write_vtk(m_isosurface_output_path /
                   fs::path{"vel_mag_" + std::to_string(iso) + "_rank_" +
                            std::to_string(m_mpi_communicator->rank()) +
                            "_time_" + std::to_string(m_iteration + 1) +
                            ".vtk"});
  }
}
//==============================================================================
}  // namespace tatooine::insitu
//==============================================================================
