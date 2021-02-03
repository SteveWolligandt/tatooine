#ifndef TATOOINE_INSITU_BOOST_MPI_H
#define TATOOINE_INSITU_BOOST_MPI_H
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
namespace boost::mpi {
//==============================================================================
namespace detail {
//==============================================================================
using coords_t    = std::vector<int>;
using rank_coords = std::pair<int, coords_t>;

/// \brief Iterator over direct (von Neumann) neighbor processes of a process
/// with a given coordinate.
class direct_neighbor_iterator
    : public iterator_facade<direct_neighbor_iterator, rank_coords,
                             forward_traversal_tag, rank_coords const&, int> {
  mutable rank_coords           current;
  cartesian_communicator const* comm;
  coords_t                      center;
  int                           index;

 public:
  direct_neighbor_iterator() : comm(nullptr), index(-1) {}

  direct_neighbor_iterator(cartesian_communicator const& comm,
                           coords_t const& coordinates, int index = 0)
      : comm(&comm), center(coordinates), index(index) {
    ensure_valid_index();
  }

 protected:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  auto dereference() const -> rank_coords const& {
    current.second = index_to_coord(index);
    current.first  = comm->rank(current.second);
    return current;
  }
  //----------------------------------------------------------------------------
  auto equal(direct_neighbor_iterator const& other) const -> bool {
    return index == other.index;
  }
  //----------------------------------------------------------------------------
  auto increment() -> void{
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
      if (!dims[i].periodic &&
          (coords[i] < 0 || coords[i] >= dims[i].size)) {
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

  auto index_to_coord(int index) const -> coords_t {
    // which dimension?
    int dim = index / 2;
    // which direction?
    int      dir = (index % 2) * 2 - 1;
    coords_t result{center};
    result[dim] += dir;
    return result;
  }

  auto max_index() const -> int { return comm->ndims() * 2; }
};

/// \brief Iterator over neighbor processes (Moore neighborhood) of a process
/// with a given coordinate.
class neighbor_iterator
    : public iterator_facade<neighbor_iterator, rank_coords,
                             forward_traversal_tag, rank_coords const&, int> {

  mutable rank_coords           current;
  cartesian_communicator const* comm;
  coords_t                      center;
  int                           index;
 public:
  neighbor_iterator() : comm{nullptr}, index{-1} {}

  neighbor_iterator(cartesian_communicator const& comm,
                    coords_t const& coordinates, int index = 0)
      : comm(&comm), center(coordinates), index(index) {
    ensure_valid_index();
  }

 protected:
  friend class boost::iterator_core_access;
  //----------------------------------------------------------------------------
  auto dereference() const ->rank_coords const& {
    current.second = index_to_coord(index);
    current.first  = comm->rank(current.second);
    return current;
  }
  //----------------------------------------------------------------------------
  auto equal(neighbor_iterator const& other) const -> bool{
    return index == other.index;
  }
  //----------------------------------------------------------------------------
  auto increment() ->void{
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
      if (!dims[i].periodic &&
          (coords[i] < 0 || coords[i] >= dims[i].size)) {
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
  auto index_to_coord(int index) const -> coords_t {
    // coordinate centered at 0
    coords_t result(comm->ndims());
    int      dim_prod = 1;
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = (index / dim_prod) % 3 - 1;
      dim_prod *= 3;
    }
    range::transform(center, result, result.begin(), std::plus<int>{});
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

using direct_neighbor_range = iterator_range<direct_neighbor_iterator>;
using neighbor_range        = iterator_range<neighbor_iterator>;

}  // end namespace detail

template <typename Coordinates>
auto neighbors(Coordinates&& coordinates, cartesian_communicator const& comm)
    -> detail::neighbor_range {
  return {detail::neighbor_iterator{comm,
                                    std::forward<Coordinates>(coordinates), 0},
          detail::neighbor_iterator{}};
}
//==============================================================================
}  // namespace boost::mpi
//==============================================================================
#endif
