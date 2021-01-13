#ifndef TATOOINE_MPI_INTERFACES_TEST_H
#define TATOOINE_MPI_INTERFACES_TEST_H
//==============================================================================
#include <mpi.h>
//==============================================================================
namespace tatooine::mpi::interfaces::test {
//==============================================================================

/// Initialize the MPI environment.
/// Call first.
extern "C" void tatooine_mpi_initialize(MPI_Fint* communicator);

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
extern "C" void tatooine_mpi_initialize_grid(
    int* gridstx, int* gridsty, int* gridstz, int* gridx, int* gridy,
    int* gridz, int* xst, int* yst, int* zst, int* xsz, int* ysz, int* zsz,
    double* dx, double* dy, double* dz, int* periodicx, int* periodicy,
    int* periodicz, int* halo_level);

/// \brief  Add variables to the dataset.
///
///         Call after tatooine_initialize_grid() and before
///         tatooine_initialize_tracking()
///
///         The data must have halos only in the directions where the current
///         processor block does not span the whole domain.
///         If a direction does have a halo, it must have exactly the size
///         of the @a halo_level parameter in tatooine_initialize_grid()
///
/// \param  mixture_fraction           scalar with halo
/// \param  prev_mixture_fraction      scalar with halo
/// \param  mixture_fraction_gradient  vector with halo
/// \param  flow_velocity              vector with halo
extern "C" void tatooine_mpi_initialize_variables(
    double* mixture_fraction, double* prev_mixture_fraction,
    double* mixture_fraction_gradient, double* flow_velocity,
    double* heat_release);

/// \brief  Set parameters for tracking.
///
///         Call before tatooine_initialize_tracking()
///
/// \param  max_init_distance        Maximum initialization distance of tracers
/// \param  seeding_error_init       Maximum deviation from tangent plane for
///                                  initial seeding of particles and ghost
///                                  particles
/// \param  target_iso_value         Iso value of mixture fraction describing
/// the
///                                  flame surface
/// \param  saving_tracers_interval  Interval at which tracers are written to
///                                  disk
/// \param  time                     Initial time
/// \param  prev_time                Initial time of virtual previous
///                                  time step
/// \param  iteration                Initial iteration number
extern "C" void tatooine_mpi_initialize_parameters(
    double* max_init_distance, double* seeding_error_init,
    double* normal_error_max, double* normal_error_min,
    double* tangential_error_max, double* tangential_error_min,
    double* target_iso_value, int* split_interval, int* saving_tracers_interval,
    double* transformation_start_time, double* transformation_interval_step,
    int* num_transformation_intervals, double* time, double* prev_time,
    int* iteration);

/// Initialize the flame front tracers and complete the initialization. Call
/// after all other initialize functions
///
/// \param  restart  Restart from saved file (1) or start from scratch (0)
extern "C" void tatooine_mpi_initialize_tracking(int* restart);

/// \brief  Update the grid variables to their new values after a simulation
/// step
///
/// \param  mixture_fraction           scalar with halo
/// \param  prev_mixture_fraction      scalar with halo
/// \param  mixture_fraction_gradient  vector with halo
/// \param  flow_velocity              vector with halo
extern "C" void tatooine_mpi_update_variables(double* mixture_fraction,
                                              double* prev_mixture_fraction,
                                              double* mixture_fraction_gradient,
                                              double* flow_velocity,
                                              double* heat_release);

extern "C" void tatooine_mpi_update(int* iteration, double* time);
//==============================================================================
}  // namespace tatooine::mpi::interfaces::test
//==============================================================================
#endif
