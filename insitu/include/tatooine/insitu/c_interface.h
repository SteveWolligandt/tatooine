#ifndef TATOOINE_INSITU_C_INTERFACE_H
#define TATOOINE_INSITU_C_INTERFACE_H
//==============================================================================
#include <mpi.h>
//==============================================================================
extern "C" {
//==============================================================================
/// Initialize the MPI environment.
/// Call first.
auto tatooine_insitu_interface_initialize_communicator(MPI_Fint* communicator)
    -> void;
//------------------------------------------------------------------------------
/// \brief  Initialize the dataset grid. Call after tatooine_initialize_mpi()
/// and
///         before tatooine_initialize_variables().
///
/// \param  gridx, gridy, gridz     global grid dimensions
/// \param  xst, yst, zst           starting indices of current
///                                 process
/// \param  xsz, ysz, zsz           number of grid points of current process
/// \param  dx, dy, dz              size of domain box
/// \param  periodicx, periodicy, periodicz     periodic boundary directions (0
///                                             for no, 1 for yes)
/// \param  halo_level              number of halo cell layers
auto tatooine_insitu_interface_initialize_grid(
    int const* global_grid_size_x, int const* global_grid_size_y,
    int const* global_grid_size_z, int const* local_starting_index_x,
    int const* local_starting_index_y, int const* local_starting_index_z,
    int const* local_grid_size_x, int const* local_grid_size_y,
    int const* local_grid_size_z, double const* domain_size_x,
    double const* domain_size_y, double const* domain_size_z,
    int const* is_periodic_x, int const* is_periodic_y,
    int const* is_periodic_z, int const* halo_level) -> void;
//------------------------------------------------------------------------------
auto tatooine_insitu_interface_initialize_velocity_x(double const* vel_x)
    -> void;
auto tatooine_insitu_interface_initialize_velocity_y(double const* vel_y)
    -> void;
auto tatooine_insitu_interface_initialize_velocity_z(double const* vel_z)
    -> void;
//------------------------------------------------------------------------------
/// \brief  Set parameters.
///
///         Call before tatooine_initialize()
///
/// \param  time                     Initial time
/// \param  prev_time                Initial time of virtual previous
///                                  time step
/// \param  iteration                Initial iteration number
void tatooine_insitu_interface_initialize_parameters(double const* time,
                                                     double const* prev_time,
                                                     int const*    iteration);
//------------------------------------------------------------------------------
/// Initialize the flame front tracers and complete the initialization. Call
/// after all other initialize functions
///
/// \param  restart  Restart from saved file (1) or start from scratch (0)
void tatooine_insitu_interface_initialize(int const* restart);
//------------------------------------------------------------------------------
/// \brief  Update the grid variables to their new values after a simulation
/// step
void tatooine_insitu_interface_update_velocity_x(double const* vel_x);
void tatooine_insitu_interface_update_velocity_y(double const* vel_y);
void tatooine_insitu_interface_update_velocity_z(double const* vel_z);
//------------------------------------------------------------------------------
void tatooine_insitu_interface_update(int const* iteration, double const* time);
//==============================================================================
}  // extern "C"
//==============================================================================
#endif
