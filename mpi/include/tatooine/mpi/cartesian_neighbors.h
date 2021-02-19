#ifndef TATOOINE_MPI_CARTESIAN_NEIGHBORS_H
#define TATOOINE_MPI_CARTESIAN_NEIGHBORS_H
//==============================================================================
#include <boost/algorithm/cxx11/all_of.hpp>
#include <boost/assert.hpp>
#include <boost/concept_check.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/concepts.hpp>
#include <boost/range/iterator_range.hpp>

//==============================================================================
namespace tatooine::mpi {
//==============================================================================
/// \brief Iterator over direct (von Neumann) neighbor processes of a process
/// with a given coordinate.
class direct_cartesian_neighbor_iterator
    : public boost::iterator_facade<
          direct_cartesian_neighbor_iterator, std::pair<int, std::vector<int>>,
          boost::forward_traversal_tag, std::pair<int, std::vector<int>> const&,
          int> {
  mutable std::pair<int, std::vector<int>>  current;
  boost::mpi::cartesian_communicator const* comm;
  std::vector<int>                          center;
  int                                       index;

 public:
  direct_cartesian_neighbor_iterator() : comm(nullptr), index(-1) {}

  direct_cartesian_neighbor_iterator(
      boost::mpi::cartesian_communicator const& comm,
      std::vector<int> const& coordinates, int index = 0)
      : comm(&comm), center(coordinates), index(index) {
    ensure_valid_index();
  }

 protected:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  auto dereference() const -> std::pair<int, std::vector<int>> const& {
    current.second = index_to_coord(index);
    current.first  = comm->rank(current.second);
    return current;
  }
  //----------------------------------------------------------------------------
  auto equal(direct_cartesian_neighbor_iterator const& other) const -> bool {
    return index == other.index;
  }
  //----------------------------------------------------------------------------
  auto increment() -> void {
    ++index;
    ensure_valid_index();
  }
  //----------------------------------------------------------------------------
  /// Check if the coordinates are valid inside the communicator. Newer MPI
  /// versions raise an error instead of returning MPI_PROC_NULL for out-of-
  /// range coordinates in nonperiodic directions, so better be safe.
  auto valid_index(int index) -> bool {
    auto const dims   = comm->topology().stl();
    auto const coords = index_to_coord(index);
    for (std::size_t i = 0; int(i) < comm->ndims(); ++i) {
      if (!dims[i].periodic && (coords[i] < 0 || coords[i] >= dims[i].size)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto ensure_valid_index() -> void {
    while (index < max_index() && !valid_index(index)) {
      ++index;
    }

    if (index >= max_index()) {
      index = -1;
    }
  }

  auto index_to_coord(int index) const -> std::vector<int> {
    // which dimension?
    int dim = index / 2;
    // which direction?
    int              dir = (index % 2) * 2 - 1;
    std::vector<int> result{center};
    result[dim] += dir;
    return result;
  }

  auto max_index() const -> int { return comm->ndims() * 2; }
};

/// \brief Iterator over neighbor processes (Moore neighborhood) of a process
/// with a given coordinate.
class cartesian_neighbor_iterator
    : public boost::iterator_facade<
          cartesian_neighbor_iterator, std::pair<int, std::vector<int>>,
          boost::forward_traversal_tag, std::pair<int, std::vector<int>> const&,
          int> {
  mutable std::pair<int, std::vector<int>>  current;
  boost::mpi::cartesian_communicator const* comm;
  std::vector<int>                          center;
  int                                       index;

 public:
  cartesian_neighbor_iterator() : comm{nullptr}, index{-1} {}

  cartesian_neighbor_iterator(boost::mpi::cartesian_communicator const& comm,
                              std::vector<int> const& coordinates,
                              int                     index = 0)
      : comm(&comm), center(coordinates), index(index) {
    ensure_valid_index();
  }

 protected:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  auto dereference() const -> std::pair<int, std::vector<int>> const& {
    current.second = index_to_coord(index);
    current.first  = comm->rank(current.second);
    return current;
  }
  //----------------------------------------------------------------------------
  auto equal(cartesian_neighbor_iterator const& other) const -> bool {
    return index == other.index;
  }
  //----------------------------------------------------------------------------
  auto increment() -> void {
    ++index;
    ensure_valid_index();
  }
  //----------------------------------------------------------------------------
  /// Check if the coordinates are valid inside the communicator. Newer MPI
  /// versions raise an error instead of returning MPI_PROC_NULL for out-of-
  /// range coordinates in nonperiodic directions, so better be safe.
  auto valid_index(int index) const -> bool {
    auto const dims   = comm->topology().stl();
    auto const coords = index_to_coord(index);
    for (std::size_t i = 0; int(i) < comm->ndims(); ++i) {
      if (!dims[i].periodic && (coords[i] < 0 || coords[i] >= dims[i].size)) {
        return false;
      }
    }
    return true;
  }
  //----------------------------------------------------------------------------
  auto ensure_valid_index() -> void {
    while (index == (max_index() - 1) / 2 ||
           (index < max_index() && !valid_index(index))) {
      ++index;
    }

    if (index >= max_index()) {
      index = -1;
    }
  }
  //----------------------------------------------------------------------------
  auto index_to_coord(int index) const -> std::vector<int> {
    // coordinate centered at 0
    std::vector<int> result(comm->ndims());
    int              dim_prod = 1;
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = (index / dim_prod) % 3 - 1;
      dim_prod *= 3;
    }
    boost::range::transform(center, result, result.begin(), std::plus<int>{});
    return result;
  }
  //----------------------------------------------------------------------------
  auto max_index() const -> int {
    double power = std::pow(3, comm->ndims());
    if (power > std::numeric_limits<int>::max()) {
      throw std::overflow_error("Communicator has too many dimensions");
    }
    return static_cast<int>(power);
  }
};
//==============================================================================
using direct_neighbor_range =
    boost::iterator_range<direct_cartesian_neighbor_iterator>;
using neighbor_range = boost::iterator_range<cartesian_neighbor_iterator>;
//==============================================================================
template <typename Coordinates>
auto cartesian_neighbors(Coordinates&&                             coordinates,
                         boost::mpi::cartesian_communicator const& comm)
    -> neighbor_range {
  return {cartesian_neighbor_iterator{
              comm, std::forward<Coordinates>(coordinates), 0},
          cartesian_neighbor_iterator{}};
}
//==============================================================================
}  // namespace tatooine::mpi
//==============================================================================
#endif
