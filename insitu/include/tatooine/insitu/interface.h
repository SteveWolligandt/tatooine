#ifndef TATOOINE_INSITU_INTERFACE_H
#define TATOOINE_INSITU_INTERFACE_H
//==============================================================================
#include <tatooine/insitu/base_interface.h>
#include <tatooine/field.h>
//==============================================================================
namespace tatooine::insitu {
//==============================================================================
struct interface : base_interface<interface> {
  using this_t   = interface;
  using parent_t = base_interface<this_t>;

  static constexpr std::string_view m_timings_fname =
      "tatooine_insitu_interface_timings.txt";
  static constexpr std::string_view m_memory_fname =
      "tatooine_insitu_interface_memory.txt";
  static constexpr std::string_view m_split_merge_fname =
      "tatooine_insitu_interface_splitmerge.txt";
  static std::filesystem::path m_output_path;
  static std::filesystem::path m_isosurface_output_path;

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
                   grid_prop_t const& x, grid_prop_t const& y,
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
  auto initialize_velocity_x(double const* vel_x) -> void;
  //----------------------------------------------------------------------------
  auto initialize_velocity_y(double const* vel_y) -> void;
  //----------------------------------------------------------------------------
  auto initialize_velocity_z(double const* vel_z) -> void;
  //----------------------------------------------------------------------------
  auto initialize_parameters(double const time, double const prev_time,
                             int const iteration) -> void;
  //----------------------------------------------------------------------------
  auto initialize(bool const restart) -> void;
  //----------------------------------------------------------------------------
  auto update_velocity_x(double const* var) -> void;
  //----------------------------------------------------------------------------
  auto update_velocity_y(double const* var) -> void;
  //----------------------------------------------------------------------------
  auto update_velocity_z(double const* var) -> void;
  //----------------------------------------------------------------------------
  auto update(int const iteration, double const time) -> void;
  //----------------------------------------------------------------------------
  auto extract_isosurfaces() -> void;
  auto extract_isosurfaces_velocity_magnitude() -> void;
};
//==============================================================================
}  // namespace tatooine::insitu
//==============================================================================
#endif
