#ifndef TATOOINE_EDGE_H
#define TATOOINE_EDGE_H
//==============================================================================
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include "grid_vertex.h"
#include "linspace.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct grid_edge
    : public std::pair<grid_vertex<Real, N>, grid_vertex<Real, N>> {
  using vertex_t = grid_vertex<Real, N>;
  using parent_t = std::pair<vertex_t, vertex_t>;
  //============================================================================
  grid_edge(const vertex_t& v0, const vertex_t& v1)
      : parent_t(std::min(v0, v1), std::max(v0, v1)) {}
  //----------------------------------------------------------------------------
  grid_edge(const grid_edge& e) : parent_t(e) {}
  //----------------------------------------------------------------------------
  grid_edge(grid_edge&& e) noexcept : parent_t(std::move(e)) {}
  //----------------------------------------------------------------------------
  auto operator=(const grid_edge& e) -> grid_edge& = default;
  auto operator=(grid_edge&& e) noexcept -> grid_edge& = default;
  //----------------------------------------------------------------------------
  ~grid_edge() = default;
  //----------------------------------------------------------------------------
  auto as_position_pair() const {
    return std::pair{*this->first, *this->second};
  }
  //----------------------------------------------------------------------------
  auto operator*() const { return as_position_pair(); }
  //----------------------------------------------------------------------------
  auto operator==(const grid_edge& other) const -> bool {
    return (this->first == other.second && this->second == other.first) ||
           (this->first == other.first && this->second == other.second);
  }
  //----------------------------------------------------------------------------
  auto operator!=(const grid_edge& other) const -> bool {
    return !operator==(other);
  }
  //----------------------------------------------------------------------------
  auto operator<(const grid_edge& other) const -> bool {
    for (size_t i = 0; i < N; ++i) {
      if (this->first.iterators[i].i() < other.first.iterators[i].i()) {
        return true;
      }
      if (this->first.iterators[i].i() > other.first.iterators[i].i()) {
        return false;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      if (this->second.iterators[i].i() < other.second.iterators[i].i()) {
        return true;
      }
      if (this->second.iterators[i].i() > other.second.iterators[i].i()) {
        return false;
      }
    }
    return false;
  }
  //----------------------------------------------------------------------------
  auto operator<=(const grid_edge& other) const -> bool {
    return operator==(other) || operator<(other);
  }
  //----------------------------------------------------------------------------
  auto operator>(const grid_edge& other) const -> bool {
    for (size_t i = 0; i < N; ++i) {
      if (this->first.iterators[i].i > other.first.iterators[i].i) {
        return true;
      }
      if (this->first.iterators[i].i < other.first.iterators[i].i) {
        return false;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      if (this->second.iterators[i].i > other.second.iterators[i].i) {
        return true;
      }
      if (this->second.iterators[i].i < other.second.iterators[i].i) {
        return false;
      }
    }
    return false;
  }
  //----------------------------------------------------------------------------
  auto operator>=(const grid_edge& other) const -> bool {
    return operator==(other) || operator>(other);
  }
  //----------------------------------------------------------------------------
  auto to_string() {
    return this->first.to_string() + ' ' + this->second.to_string();
  }
};
//==============================================================================
template <typename Real, size_t N>
inline auto operator<<(std::ostream& out, const grid_edge<Real, N>& e)
    -> auto& {
  out << e.first << " - " << e.second;
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
