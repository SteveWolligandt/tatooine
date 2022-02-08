#ifndef TATOOINE_INSITU_INTERFACE_H
#define TATOOINE_INSITU_INTERFACE_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/insitu/base_interface.h>
//==============================================================================
namespace tatooine::insitu {
//==============================================================================
struct interface : base_interface<interface> {
  using this_type   = interface;
  using parent_t = base_interface<this_type>;
  using pos_type    = vec3;

  using scalar_arr_t = non_owning_multidim_array<double, x_fastest>;
  using grid_prop_t =
      uniform_rectilinear_grid<double, 3>::typed_property_impl_t<scalar_arr_t>;
  using tracer_t           = std::pair<size_t, pos_type>;
  using tracer_container_t = std::vector<tracer_t>;

  struct velocity_field : vectorfield<velocity_field, double, 3> {
    using this_type   = velocity_field;
    using parent_t = vectorfield<this_type, double, 3>;
    using parent_t::pos_type;
    using parent_t::real_type;
    using parent_t::tensor_type;

    uniform_rectilinear_grid<double, 3> const& m_worker_halo_grid;
    grid_prop_t const&             m_x;
    grid_prop_t const&             m_y;
    grid_prop_t const&             m_z;

    velocity_field(uniform_rectilinear_grid<double, 3> const& worker_halo_grid,
                   grid_prop_t const& x, grid_prop_t const& y,
                   grid_prop_t const& z)
        : m_worker_halo_grid{worker_halo_grid}, m_x{x}, m_y{y}, m_z{z} {}

    auto evaluate(pos_type const& x, real_type const /*t*/) const
        -> tensor_type override {
      return {m_x.sampler<interpolation::linear>()(x),
              m_y.sampler<interpolation::linear>()(x),
              m_z.sampler<interpolation::linear>()(x)};
    }
    auto in_domain(pos_type const& x, real_type const /*t*/) const -> bool override {
      return m_worker_halo_grid.in_domain(x(0), x(1), x(2));
    }
  };

  static constexpr std::string_view m_timings_fname =
      "tatooine_insitu_interface_timings.txt";
  static constexpr std::string_view m_memory_fname =
      "tatooine_insitu_interface_memory.txt";
  static constexpr std::string_view m_split_merge_fname =
      "tatooine_insitu_interface_splitmerge.txt";
  static filesystem::path m_output_path;
  static filesystem::path m_isosurface_output_path;
  static filesystem::path m_tracers_output_path;
  static filesystem::path m_tracers_tmp_path;

  grid_prop_t *m_velocity_x, *m_velocity_y, *m_velocity_z;

  std::ofstream m_timings_file;

  size_t                          m_num_tracers = 10;
  tracer_container_t              m_tracers;
  std::unique_ptr<velocity_field> m_velocity_field;

  //============================================================================
  // Interface Functions
  //============================================================================
  auto initialize_grid(int global_grid_size_x, int global_grid_size_y,
                       int global_grid_size_z, int local_starting_index_x,
                       int local_starting_index_y, int local_starting_index_z,
                       int local_grid_size_x, int local_grid_size_y,
                       int local_grid_size_z, double domain_size_x,
                       double domain_size_y, double domain_size_z,
                       int is_periodic_x, int is_periodic_y, int is_periodic_z,
                       int halo_level) -> void;
  //----------------------------------------------------------------------------
  auto initialize_velocity_x(double const* vel_x) -> void;
  //----------------------------------------------------------------------------
  auto initialize_velocity_y(double const* vel_y) -> void;
  //----------------------------------------------------------------------------
  auto initialize_velocity_z(double const* vel_z) -> void;
  //----------------------------------------------------------------------------
  auto initialize_parameters(double time, double prev_time, int iteration)
      -> void;
  //----------------------------------------------------------------------------
  auto initialize(bool restart) -> void;
  //----------------------------------------------------------------------------
  auto update_velocity_x(double const* var) -> void;
  //----------------------------------------------------------------------------
  auto update_velocity_y(double const* var) -> void;
  //----------------------------------------------------------------------------
  auto update_velocity_z(double const* var) -> void;
  //----------------------------------------------------------------------------
  auto update(int iteration, double time) -> void;
  //----------------------------------------------------------------------------
  /// Updates the position x to the advected position.
  /// \param x initial positions (gets changed)
  /// \return advected position
  auto advect_tracer(pos_type& pos) -> bool;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// Updates the positions xs to the advected positions
  /// \param xs initial positions (get changed)
  /// \return advected positions
  auto advect_tracers() -> void;
  //----------------------------------------------------------------------------
  auto create_tracer_vtk() -> void;
  //----------------------------------------------------------------------------
  auto extract_isosurfaces() -> void;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto extract_isosurfaces_velocity_magnitude() -> void;
};
//==============================================================================
}  // namespace tatooine::insitu
//==============================================================================
#endif
