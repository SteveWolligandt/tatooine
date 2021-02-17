#ifndef TATOOINE_INSITU_MPI_PROGRAM_H
#define TATOOINE_INSITU_MPI_PROGRAM_H
//==============================================================================
#include <mpi.h>
#include <tatooine/grid.h>
#include <tatooine/insitu/boost_mpi.h>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/serialization/optional.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/variant.hpp>
#include <boost/serialization/vector.hpp>
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
  static auto get(int& argc, char** argv) -> mpi_program& {
    static mpi_program p{argc, argv};
    return p;
  }
  //============================================================================
  // Members
  //============================================================================
  MPI_Comm                                            m_communicator;
  MPI_Fint                                            m_communicator_fint;
  std::unique_ptr<boost::mpi::cartesian_communicator> m_mpi_communicator;
  int                                                 m_rank           = 0;
  int                                                 m_num_processes  = 0;
  int                                                 m_num_dimensions = 0;

  std::unique_ptr<int[]>    m_num_processes_per_dimension;
  std::unique_ptr<int[]>    m_periods;
  std::unique_ptr<size_t[]> m_global_grid_size;

  std::unique_ptr<int[]> m_process_periods;
  std::unique_ptr<int[]> m_process_coords;
  std::unique_ptr<int[]> m_process_size;

  std::unique_ptr<size_t[]> m_process_begin_indices;
  std::unique_ptr<size_t[]> m_process_end_indices;
  std::unique_ptr<bool[]>   m_is_single_cell;

  //============================================================================
  // Ctor
  //============================================================================
  mpi_program(int& argc, char** argv);

 public:
  mpi_program(mpi_program const&) = delete;
  mpi_program(mpi_program&&)      = delete;
  auto operator=(mpi_program const&) -> mpi_program& = delete;
  auto operator=(mpi_program&&) -> mpi_program& = delete;

  /// Destructor terminating mpi
  ~mpi_program();
  //============================================================================
  // Methods
  //============================================================================
  auto rank() const { return m_rank; }
  auto num_dimensions() const { return m_num_dimensions; }
  auto num_processes() const { return m_num_processes; }
  auto global_grid_size() const -> auto const& { return m_global_grid_size; }
  auto global_grid_size(size_t i) const { return m_global_grid_size[i]; }

  auto periods() const -> auto const& { return m_periods; }
  auto is_periodic(size_t const i) const -> auto const& { return m_periods[i]; }
  auto process_periods() const -> auto const& { return m_process_periods; }
  auto process_period(size_t const i) const { return m_process_periods[i]; }

  auto num_processes_per_dimensions() const -> auto const& {
    return m_num_processes_per_dimension;
  }
  auto num_processes_in_dimension(size_t const i) const -> auto const& {
    return m_num_processes_per_dimension[i];
  }

  auto process_begin_indices() const -> auto const& {
    return m_process_begin_indices;
  }
  auto process_begin(size_t const i) const {
    return m_process_begin_indices[i];
  }
  auto process_end_indices() const -> auto const& {
    return m_process_end_indices;
  }
  auto process_end(size_t const i) const { return m_process_end_indices[i]; }
  auto process_size() const -> auto const& { return m_process_size; }
  auto process_size(size_t const i) const { return m_process_size[i]; }
  auto is_single_cell() const -> auto const& { return m_is_single_cell; }
  auto is_single_cell(size_t const i) const -> auto const& {
    return m_is_single_cell[i];
  }

  auto communicator() const -> auto { return m_communicator; }
  auto communicator_fint() const -> auto { return m_communicator_fint; }

  //----------------------------------------------------------------------------
  template <typename... Dimensions>
  auto init_communicator(grid<Dimensions...> const& g) -> void {
    auto global_grid_dimensions =
        std::unique_ptr<size_t[]>{new size_t[g.num_dimensions()]};
    for (unsigned int i = 0; i < g.num_dimensions(); ++i) {
      global_grid_dimensions[i] = g.size(i);
    }
    init_communicator(g.num_dimensions(), std::move(global_grid_dimensions));
  }
  //----------------------------------------------------------------------------
  template <typename... GlobalGridDimensions,
            enable_if<is_arithmetic<GlobalGridDimensions...>> = true>
  auto init_communicator(GlobalGridDimensions const... ggd) -> void {
    constexpr auto num_dimensions = sizeof...(GlobalGridDimensions);
    init_communicator(
        num_dimensions,
        std::unique_ptr<size_t[]>{new size_t[]{static_cast<size_t>(ggd)...}});
  }
  //----------------------------------------------------------------------------
  auto init_communicator(int                         num_dimensions,
                         std::unique_ptr<size_t[]>&& global_grid_dimensions)
      -> void;
  //----------------------------------------------------------------------------
  template <typename T>
  auto gather(T const& in, int const root) const {
    std::vector<T> out;
    boost::mpi::gather(*m_mpi_communicator, in, out, root);
    return out;
  }
  //----------------------------------------------------------------------------
  template <typename T, typename ReceiveHandler>
  auto all_gather(std::vector<T> const& outgoing,
                  ReceiveHandler&&      receive_handler) const {
    std::vector<std::vector<T>> received_data;
    boost::mpi::all_gather(*m_mpi_communicator, outgoing, received_data);

    for (auto const& rec : received_data) {
      boost::for_each(rec, receive_handler);
    }
  }
  /// \brief Communicate a number of elements with all neighbor processes
  /// \details Sends a number of \p outgoing elements to all neighbors in the
  ///      given \p communicator. Receives a number of elements of the same type
  ///      from all neighbors. Calls the \p receive_handler for each received
  ///      element.
  ///
  /// \param outgoing Collection of elements to send to all neighbors
  /// \param comm Communicator for the communication
  /// \param receive_handler Functor that is called with each received element
  ///
  template <typename T, typename ReceiveHandler>
  auto gather_neighbors(std::vector<T> const& outgoing,
                        ReceiveHandler&&      receive_handler) -> void {
    namespace mpi = boost::mpi;
    auto sendreqs = std::vector<mpi::request>{};
    auto recvreqs = std::map<int, mpi::request>{};
    auto incoming = std::map<int, std::vector<T>>{};

    for (auto const& [rank, coords] : mpi::neighbors(
             m_mpi_communicator->coordinates(m_mpi_communicator->rank()),
             *m_mpi_communicator)) {
      incoming[rank] = std::vector<T>{};
      recvreqs[rank] = m_mpi_communicator->irecv(rank, rank, incoming[rank]);
      sendreqs.push_back(m_mpi_communicator->isend(
          rank, m_mpi_communicator->rank(), outgoing));
    }

    // wait for and handle receive requests
    while (!recvreqs.empty()) {
      for (auto& [key, req] : recvreqs) {
        if (req.test()) {
          boost::for_each(incoming[key], receive_handler);
          recvreqs.erase(key);
          break;
        }
      }
    }

    // wait for send requests to finish
    mpi::wait_all(begin(sendreqs), end(sendreqs));
  }
};
//==============================================================================
}  // namespace tatooine::insitu
//==============================================================================
#endif
