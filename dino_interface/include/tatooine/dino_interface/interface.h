#ifndef TATOOINE_DINO_INTERFACE_INTERFACE_H
#define TATOOINE_DINO_INTERFACE_INTERFACE_H
//==============================================================================
#include <mpi.h>
//==============================================================================
extern "C" {
/// Initialize the MPI environment.
/// Call first.
auto tatooine_dino_interface_initialize_communicator(MPI_Fint* communicator)
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
auto tatooine_dino_interface_initialize_grid(
    int const* global_grid_size_x, int const* global_grid_size_y,
    int const* global_grid_size_z, int const* local_starting_index_x,
    int const* local_starting_index_y, int const* local_starting_index_z,
    int const* local_grid_size_x, int const* local_grid_size_y,
    int const* local_grid_size_z, double const* domain_size_x,
    double const* domain_size_y, double const* domain_size_z,
    int const* is_periodic_x, int const* is_periodic_y,
    int const* is_periodic_z, int const* halo_level) -> void;
//------------------------------------------------------------------------------
/// \brief  Add variables to the dataset.
///
///         Call after tatooine_initialize_grid() and before
///         tatooine_initialize()
///
///         The data must have halos only in the directions where the current
///         processor block does not span the whole domain.
///         If a direction does have a halo, it must have exactly the size
///         of the @a halo_level parameter in tatooine_initialize_grid()
auto tatooine_dino_interface_initialize_variable(char const*   name,
                                                 int const*    num_components,
                                                 double const* variable)
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
void tatooine_dino_interface_initialize_parameters(double const* time,
                                                   double const* prev_time,
                                                   int const*    iteration);
//------------------------------------------------------------------------------
/// Initialize the flame front tracers and complete the initialization. Call
/// after all other initialize functions
///
/// \param  restart  Restart from saved file (1) or start from scratch (0)
void tatooine_dino_interface_initialize(int const* restart);
//------------------------------------------------------------------------------
/// \brief  Update the grid variables to their new values after a simulation
/// step
void tatooine_dino_interface_update_variable(char const*   name,
                                             int const*    num_components,
                                             double const* variable);
//------------------------------------------------------------------------------
void tatooine_dino_interface_update(int const* iteration, double const* time);
}
#endif