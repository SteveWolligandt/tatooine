#ifndef TATOOINE_MPI_INTERFACES_TEST_H
#define TATOOINE_MPI_INTERFACES_TEST_H
//==============================================================================
#include <mpi.h>
//==============================================================================
/// Initialize the MPI environment.
/// Call first.
extern "C" void tatooine_dino_initialize_communicator(MPI_Fint* communicator);
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
extern "C" void tatooine_dino_initialize_grid(
    int const* gridx, int const* gridy, int const* gridz,
    int const* xst, int const* yst, int const* zst,
    int const* xsz, int const* ysz, int const* zsz,
    double const* dx, double const* dy, double const* dz,
    int const* periodicx, int const* periodicy, int const* periodicz,
    int const* halo_level);
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
extern "C" void tatooine_dino_initialize_variable(char const*   name,
                                                  int const*    num_components,
                                                  double const* variable);
//------------------------------------------------------------------------------
/// \brief  Set parameters.
///
///         Call before tatooine_initialize()
///
/// \param  time                     Initial time
/// \param  prev_time                Initial time of virtual previous
///                                  time step
/// \param  iteration                Initial iteration number
extern "C" void tatooine_dino_initialize_parameters(double const* time,
                                                    double const* prev_time,
                                                    int const*    iteration);
//------------------------------------------------------------------------------
/// Initialize the flame front tracers and complete the initialization. Call
/// after all other initialize functions
///
/// \param  restart  Restart from saved file (1) or start from scratch (0)
extern "C" void tatooine_dino_initialize(int const* restart);
//------------------------------------------------------------------------------
/// \brief  Update the grid variables to their new values after a simulation
/// step
extern "C" void tatooine_dino_update_variable(char const*   name,
                                              int const*    num_components,
                                              double const* variable);
//------------------------------------------------------------------------------
extern "C" void tatooine_dino_update(int const* iteration, double const* time);
#endif
