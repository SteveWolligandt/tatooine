#ifndef TATOOINE_INSITU_MPI_PROGRAM_H
#define TATOOINE_INSITU_MPI_PROGRAM_H
//==============================================================================
#include <mpi.h>
#include <tatooine/grid.h>
#include <memory>
//==============================================================================
namespace tatooine::insitu {
//==============================================================================
struct mpi_program {
 public:
  //============================================================================
  // Singleton Getter
  //============================================================================
  /// Creates the mpi program.
  /// Store the returned object as reference!
  template <typename... Dimensions>
  static auto get(int const argc, char const** argv,
                  grid<Dimensions...> const& g) -> auto& {
    auto global_grid_dimensions =
      std::unique_ptr<size_t[]> {new size_t[g.num_dimensions()]};
    for (unsigned int i = 0; i < g.num_dimensions(); ++i) {
      global_grid_dimensions[i] = g.size(i);
    }
    return get(argc, argv, g.num_dimensions(),
               std::move(global_grid_dimensions));
  }
 private:
  static auto get(int const argc, char const** argv, int const num_dimensions,
                  std::unique_ptr<size_t[]>&& global_grid_dimensions) -> auto& {
    static mpi_program p{argc, argv, num_dimensions,
                         std::move(global_grid_dimensions)};
    return p;
  }
  //============================================================================
  // Members
  //============================================================================
  MPI_Comm m_communicator;
  MPI_Fint m_communicator_fint;
  int      m_rank           = 0;
  int      m_num_processes  = 0;
  int      m_num_dimensions = 0;

  std::unique_ptr<int[]> m_dimensions;
  std::unique_ptr<int[]> m_periods;
  std::unique_ptr<size_t[]> m_global_grid_dimensions;

  std::unique_ptr<int[]> m_num_processes_per_dimension;
  std::unique_ptr<int[]> m_process_periods;
  std::unique_ptr<int[]> m_process_coords;
  std::unique_ptr<int[]> m_process_dims;

  std::unique_ptr<size_t[]> m_process_begin_indices;
  std::unique_ptr<size_t[]> m_process_end_indices;

  //============================================================================
  // Ctor
  //============================================================================
  mpi_program(int argc, char** argv, int num_dimensions,
              std::unique_ptr<size_t[]>&& global_grid_dimensions) :
      m_num_dimensions{num_dimensions},
      m_dimensions{new int[m_num_dimensions]},
      m_periods{new int[m_num_dimensions]},
      m_global_grid_dimensions{std::move(global_grid_dimensions},
      m_num_processes_per_dimension{new int[m_num_dimensions]},
      m_process_periods{new int[m_num_dimensions]},
      m_process_coords{new int[m_num_dimensions]},
      m_process_dims{new int[m_num_dimensions]},
      m_process_begin_indices{new int[m_num_dimensions]}
      m_process_end_indices{new int[m_num_dimensions]}
 {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &m_num_processes);

    MPI_Dims_create(m_num_processes, m_num_dimensions, m_dimensions.get());
    MPI_Cart_create(MPI_COMM_WORLD, m_num_dimensions, m_dimensions.get(),
                    m_periods.get(), true,
                    &m_communicator);

    MPI_Comm_set_errhandler(m_communicator, MPI_ERRORS_RETURN);

    MPI_Cartdim_get(m_communicator, &m_num_dimensions);

    MPI_Cart_get(m_communicator, m_num_dimensions, m_num_processes_per_dimension.get(),
                 m_process_periods.get(), m_process_coords.get());

    m_communicator_fint = MPI_Comm_c2f(m_communicator);

    for (int i = 0; i < m_num_dimensions; ++i) {
      m_process_dims[i] =
        m_global_grid_size[i] / m_num_processes_per_dimension[i];

      m_process_begin_indices[i] = m_process_dims * m_process_coords[i];
      m_process_begin[i] = m_process_dims[i] *
                           m_process_coords[i];
      if (m_process_coords[i] == m_process_num_dimensions[i] - 1) {
        m_process_end_indices[i] = m_global_grid_size[i];
      } else {
        m_process_end_indices[i] = m_process_begin_indices[i] +
                                   m_process_dims[i];
      }
    }
  }
 public:
  mpi_program(mpi_program const&) = delete;
  mpi_program(mpi_program &&) = delete;
  auto operator=(mpi_program const&) -> mpi_program& = delete;
  auto operator=(mpi_program &&) -> mpi_program& = delete;

  /// Destructor terminating mpi
  ~mpi_program() {
    MPI_Finalize();
  }
  //============================================================================
  // Methods
  //============================================================================
  auto rank() const {return m_rank;}
  auto num_dimensions() const { return m_num_dimensions;}
  auto num_processes() const { return m_num_processes;}
  auto global_grid_dimensions() const -> auto const& {
    return m_global_grid_dimensions;
  }

  auto periods() const -> auto const& { return m_periods; }
  auto process_periods() const -> auto const& { return m_process_periods; }

  auto dimensions() const -> auto const& { return m_dimensions; }
  auto num_processes_per_dimensions() const -> auto const& {
    return m_num_processes_per_dimension;
  }

  auto communicator() const -> auto { return m_communicator; }
  auto communicator_fint() const -> auto { return m_communicator_fint; }


};
//==============================================================================
} // namespace tatooine::insitu
//==============================================================================
