#include <tatooine/insitu/mpi_program.h>
//==============================================================================
namespace tatooine::insitu {
//==============================================================================
mpi_program::mpi_program(int& argc, char** argv) {
  int ret = MPI_SUCCESS;
  ret     = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    throw std::runtime_error{
        "[MPI_Init]\nThis error class is associated with an error code that "
        "indicates that an attempt was made to call MPI_INIT a second time. "
        "MPI_INIT may only be called once in a program."};
  }
  ret = MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  if (ret == MPI_ERR_COMM) {
    throw std::runtime_error{
        "[MPI_Comm_set_errhandler]\nInvalid communicator. A common error "
        "is "
        "to use a null communicator "
        "in a call (not even allowed in MPI_Comm_rank)."};
  } else if (ret == MPI_ERR_OTHER) {
    throw std::runtime_error{
        "[MPI_Comm_set_errhandler]\nOther error; use MPI_Error_string to get "
        "more information about this error code."};
  }
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
  if (ret == MPI_ERR_OTHER) {
    throw std::runtime_error{
        "[MPI_Comm_rank]\nInvalid communicator. A common error is to use a "
        "null communicator in a call (not even allowed in MPI_Comm_rank)."};
  }
  ret = MPI_Comm_size(MPI_COMM_WORLD, &m_num_processes);
  if (ret == MPI_ERR_COMM) {
    throw std::runtime_error{
        "[MPI_Comm_size]\nInvalid communicator. A common error is to use a "
        "null communicator in a call (not even allowed in MPI_Comm_rank)."};
  } else if (ret == MPI_ERR_OTHER) {
    throw std::runtime_error{
        "[MPI_Comm_size]\nInvalid argument. Some argument is invalid and is "
        "not identified by a specific error class."};
  }
}
//------------------------------------------------------------------------------
mpi_program::~mpi_program() { MPI_Finalize(); }
//------------------------------------------------------------------------------
auto mpi_program::init_communicator(
    int num_dimensions, std::unique_ptr<size_t[]>&& global_grid_dimensions)
    -> void {
  m_num_dimensions   = num_dimensions;
  m_global_grid_size = std::move(global_grid_dimensions);
  m_num_processes_per_dimension =
      std::unique_ptr<int[]>{new int[m_num_dimensions]};
  for (int i = 0; i < m_num_dimensions; ++i) {
    m_num_processes_per_dimension[i] = 0;
  }
  m_periods = std::unique_ptr<int[]>{new int[m_num_dimensions]};
  for (int i = 0; i < m_num_dimensions; ++i) {
    m_periods[i] = 0;
  }
  m_process_periods = std::unique_ptr<int[]>{new int[m_num_dimensions]};
  m_process_coords  = std::unique_ptr<int[]>{new int[m_num_dimensions]};
  m_process_size    = std::unique_ptr<int[]>{new int[m_num_dimensions]};
  m_process_begin_indices =
      std::unique_ptr<size_t[]>{new size_t[m_num_dimensions]};
  m_process_end_indices =
      std::unique_ptr<size_t[]>{new size_t[m_num_dimensions]};
  m_is_single_cell = std::unique_ptr<bool[]>{new bool[m_num_dimensions]};

  int ret = 0;

  MPI_Dims_create(m_num_processes, m_num_dimensions,
                  m_num_processes_per_dimension.get());
  if (ret != MPI_SUCCESS) {
    throw std::runtime_error{"[MPI_Dims_create]\nThis should not happen."};
  }
  ret = MPI_Cart_create(MPI_COMM_WORLD, m_num_dimensions,
                        m_num_processes_per_dimension.get(), m_periods.get(),
                        true, &m_communicator);
  if (ret != MPI_SUCCESS) {
    if (ret == MPI_ERR_TOPOLOGY) {
      throw std::runtime_error{
          "[MPI_Cart_create]\nInvalid topology. Either there is no topology "
          "associated with this communicator, or it is not the correct "
          "type."};
    } else if (ret == MPI_ERR_DIMS) {
      throw std::runtime_error{
          "[MPI_Cart_create]\nInvalid dimension argument. A dimension "
          "argument is null or its length is less than or equal to zero."};
    } else if (ret == MPI_ERR_ARG) {
      throw std::runtime_error{
          "[MPI_Cart_create]\nInvalid argument. Some argument is invalid and "
          "is not identified by a specific error class."};
    }
  }

  MPI_Comm_set_errhandler(m_communicator, MPI_ERRORS_RETURN);

  MPI_Cartdim_get(m_communicator, &m_num_dimensions);

  MPI_Cart_get(m_communicator, m_num_dimensions,
               m_num_processes_per_dimension.get(), m_process_periods.get(),
               m_process_coords.get());

  m_communicator_fint = MPI_Comm_c2f(m_communicator);

  m_mpi_communicator = std::unique_ptr<boost::mpi::cartesian_communicator>{
      new boost::mpi::cartesian_communicator{MPI_Comm_f2c(m_communicator_fint),
                                             boost::mpi::comm_attach}};
  for (int i = 0; i < m_num_dimensions; ++i) {
    m_is_single_cell[i] = m_num_processes_per_dimension[i] == 1;
    m_process_size[i] =
        m_global_grid_size[i] / m_num_processes_per_dimension[i];

    m_process_begin_indices[i] = m_process_size[i] * m_process_coords[i];
    if (m_process_coords[i] == m_num_processes_per_dimension[i] - 1) {
      m_process_end_indices[i] = m_global_grid_size[i];
      m_process_size[i] = m_process_end_indices[i] - m_process_begin_indices[i];
    } else {
      m_process_end_indices[i] = m_process_begin_indices[i] + m_process_size[i];
    }
  }
}
//==============================================================================
}  // namespace tatooine::insitu
//==============================================================================
