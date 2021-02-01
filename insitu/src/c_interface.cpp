#include <tatooine/insitu/c_interface.h>
#include <tatooine/insitu/interface.h>
//==============================================================================
auto tatooine_insitu_interface_initialize_communicator(MPI_Fint* communicator)
    -> void {
  tatooine::insitu::interface::get().initialize_communicator(
      *communicator);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_initialize_grid(
    int const* global_grid_size_x, int const* global_grid_size_y,
    int const* global_grid_size_z, int const* local_starting_index_x,
    int const* local_starting_index_y, int const* local_starting_index_z,
    int const* local_grid_size_x, int const* local_grid_size_y,
    int const* local_grid_size_z, double const* domain_size_x,
    double const* domain_size_y, double const* domain_size_z,
    int const* is_periodic_x, int const* is_periodic_y,
    int const* is_periodic_z, int const* halo_level) -> void {
  tatooine::insitu::interface::get().initialize_grid(
      *global_grid_size_x, *global_grid_size_y, *global_grid_size_z,
      *local_starting_index_x, *local_starting_index_y, *local_starting_index_z,
      *local_grid_size_x, *local_grid_size_y, *local_grid_size_z,
      *domain_size_x, *domain_size_y, *domain_size_z, *is_periodic_x,
      *is_periodic_y, *is_periodic_z, *halo_level);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_initialize_velocity_x(double const* vel_x)
    -> void {
  tatooine::insitu::interface::get().initialize_velocity_x(vel_x);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_initialize_velocity_y(double const* vel_y)
    -> void {
  tatooine::insitu::interface::get().initialize_velocity_y(vel_y);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_initialize_velocity_z(double const* vel_z)
    -> void {
  tatooine::insitu::interface::get().initialize_velocity_z(vel_z);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_initialize_parameters(double const* time,
                                                   double const* prev_time,
                                                   int const*    iteration)
    -> void {
  tatooine::insitu::interface::get().initialize_parameters(
      *time, *prev_time, *iteration);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_initialize(int const* restart) -> void {
  tatooine::insitu::interface::get().initialize(*restart);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_update_velocity_x(double const* vel_x) -> void {
  tatooine::insitu::interface::get().update_velocity_x(vel_x);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_update_velocity_y(double const* vel_y) -> void {
  tatooine::insitu::interface::get().update_velocity_y(vel_y);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_update_velocity_z(double const* vel_z) -> void {
  tatooine::insitu::interface::get().update_velocity_z(vel_z);
}
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_update(int const* iteration, double const* time)
    -> void {
  tatooine::insitu::interface::get().update(*iteration, *time);
}
