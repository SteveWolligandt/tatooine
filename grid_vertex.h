#ifndef TATOOINE_VERTEX_H
#define TATOOINE_VERTEX_H

#include <cassert>
#include <cstddef>
#include <functional>
#include <utility>
#include "linspace.h"
#include "tensor.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <size_t n, typename real_t>
class grid;
template <size_t n, typename real_t>
struct grid_vertex_iterator;

//==============================================================================

template <size_t n, typename real_t>
struct grid_vertex {
  using grid_t          = grid<n, real_t>;
  using linspace_iter_t = typename linspace<real_t>::iterator;
  using iterator        = grid_vertex_iterator<n, real_t>;

  //==========================================================================
  std::array<linspace_iter_t, n> iterators;

  //==========================================================================
  grid_vertex(const grid_vertex& other) : iterators{other.iterators} {}

  //--------------------------------------------------------------------------
  grid_vertex(grid_vertex&& other) : iterators{std::move(other.iterators)} {}

  //--------------------------------------------------------------------------
 private:
  template <typename... pos_ts,  size_t... Is>
  grid_vertex(const grid_t& grid, std::index_sequence<Is...>, pos_ts... pos)
   : iterators{linspace_iter_t{&grid.dimension(Is), std::size_t(pos)}...} {}

 public:
  template <typename... pos_ts,  size_t... Is>
  grid_vertex(const grid_t& grid, pos_ts... pos)
    : grid_vertex{grid, std::make_index_sequence<n>{}, pos...} {
    static_assert(sizeof...(pos_ts) == n);
    static_assert((std::is_integral_v<pos_ts> && ...));
  }

  //--------------------------------------------------------------------------
  template <typename... Its>
  grid_vertex(linspace_iter_t head_it, const Its&... tail_it)
      : iterators{head_it, tail_it...} {
    static_assert(sizeof...(Its) == n - 1,
                  "number of linspace iterators does not match n");
  }

  //--------------------------------------------------------------------------
  auto& operator=(const grid_vertex<n, real_t>& other) {
    for (size_t i = 0; i < n; ++i) iterators[i] = other.iterators[i];
    return *this;
  }

  //--------------------------------------------------------------------------
  auto& operator=(grid_vertex<n, real_t>&& other) {
    for (size_t i = 0; i < n; ++i) iterators[i] = other.iterators[i];
    return *this;
  }

  //--------------------------------------------------------------------------
  auto& operator++() {
    ++iterators.front();
    for (size_t i = 0; i < n - 1; ++i) {
      if (iterators[i] == iterators[i].get().end()) {
        iterators[i].to_begin();
        ++iterators[i + 1];
      }
    }
    return *this;
  }

  //--------------------------------------------------------------------------
  auto& operator--() {
    for (size_t i = 0; i < n; ++i)
      if (iterators[i] == iterators[i].get().begin()) {
        iterators[i].to_end();
        --iterators[i];
      } else {
        --iterators[i];
        break;
      }
    return *this;
  }

  auto&       operator[](size_t i) { return iterators[i]; }
  const auto& operator[](size_t i) const { return iterators[i]; }

  //--------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto to_vec(std::index_sequence<Is...> /*is*/) const {
    return tensor<real_t, n>{*iterators[Is]...};
  }

  template <size_t... Is>
  constexpr auto to_array(std::index_sequence<Is...> /*is*/) const {
    return std::array<real_t, n>{*iterators[Is]...};
  }

  template <size_t... Is>
  constexpr auto to_indices(std::index_sequence<Is...> /*is*/) const {
    return std::array<size_t, n>{iterators[Is].i()...};
  }

 public:
  constexpr auto to_vec() const {
    return to_vec(std::make_index_sequence<n>());
  }

  constexpr auto to_array() const {
    return to_array(std::make_index_sequence<n>());
  }

  constexpr auto to_indices() const {
    return to_indices(std::make_index_sequence<n>());
  }

  constexpr auto global_index() const {
    size_t idx = 0;
    size_t factor = 1;
    for (const auto& it : iterators) {
      idx += factor * it.i();
      factor *= it.get().size();
    }
    return idx;
  }

  constexpr auto operator*() const { return to_vec(); }

  //--------------------------------------------------------------------------

  constexpr bool operator==(const grid_vertex& other) const {
    for (size_t i = 0; i < n; ++i) {
      if (iterators[i] != other.iterators[i]) { return false; }
    }
    return true;
  }

  //--------------------------------------------------------------------------

  constexpr bool operator<(const grid_vertex& other) const {
    for (size_t i = 0; i < n; ++i) {
      if (iterators[i].i() < other.iterators[i].i()) return true;
      if (iterators[i].i() > other.iterators[i].i()) return false;
    }
    return false;
  }

  //--------------------------------------------------------------------------

  constexpr bool operator>(const grid_vertex& other) const {
    return !operator<(other);
  }

  //--------------------------------------------------------------------------

  constexpr bool operator<=(const grid_vertex& other) const {
    return operator==(other) || operator<(other);
  }

  //--------------------------------------------------------------------------

  constexpr bool operator>=(const grid_vertex& other) const {
    return operator==(other) || operator>(other);
  }

  //--------------------------------------------------------------------------

  constexpr bool operator!=(const grid_vertex& other) const {
    return !operator==(other);
  }

  auto to_string() const {
    std::string str;
    str += "[ ";
    for (const auto& i : iterators) str += std::to_string(i.i()) + ' ';
    str += "]";
    return str;
  }
};

//==============================================================================

template <size_t n, typename real_t>
struct grid_vertex_iterator {
  using difference_type   = size_t;
  using value_type        = grid_vertex<n, real_t>;
  using pointer           = value_type*;
  using reference         = value_type&;
  using iterator_category = std::bidirectional_iterator_tag;

  grid_vertex<n, real_t> v;
  auto&                  operator*() { return v; }
  auto&                  operator++() {
    ++v;
    return *this;
  }
  auto& operator--() {
    --v;
    return *this;
  }
  auto operator==(const grid_vertex_iterator& other) { return v == other.v; }
  auto operator!=(const grid_vertex_iterator& other) { return v != other.v; }
};

//==============================================================================

template <size_t n, typename real_t>
inline auto& operator<<(std::ostream& out, const grid_vertex<n, real_t>& v) {
  out << "[ ";
  for (const auto& i : v.iterators) out << i.i() << ' ';
  out << "]";
  return out;
}

template <size_t n, typename real_t>
auto next(grid_vertex_iterator<n, real_t> it, size_t num) {
  for (size_t i = 0; i < num; ++i) ++it;
  return it;
}

template <size_t n, typename real_t>
auto prev(grid_vertex_iterator<n, real_t> it, size_t num) {
  for (size_t i = 0; i < num; ++i) --it;
  return it;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
