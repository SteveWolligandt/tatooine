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

template <size_t n, typename real_t>
struct grid_edge
    : public std::pair<grid_vertex<n, real_t>, grid_vertex<n, real_t>> {
  using vertex_t = grid_vertex<n, real_t>;
  using parent_t = std::pair<vertex_t, vertex_t>;

  //============================================================================
  grid_edge(const vertex_t& v0, const vertex_t& v1)
      : parent_t(std::min(v0, v1), std::max(v0, v1)) {}

  //----------------------------------------------------------------------------
  grid_edge(const grid_edge& e) : parent_t(e) {}

  //----------------------------------------------------------------------------
  grid_edge(grid_edge&& e) : parent_t(std::move(e)) {}

  //----------------------------------------------------------------------------
  auto& operator=(const grid_edge& e) {
    parent_t::operator=(e);
    return *this;
  }

  //----------------------------------------------------------------------------
  auto& operator=(grid_edge&& e) {
    parent_t::operator=(std::move(e));
    return *this;
  }

  //----------------------------------------------------------------------------
  auto operator*() { return std::pair{*this->first, *this->second}; }

  //----------------------------------------------------------------------------
  bool operator==(const grid_edge& other) const {
    return (this->first == other.second && this->second == other.first) ||
           (this->first == other.first && this->second == other.second);
  }

  //----------------------------------------------------------------------------
  bool operator!=(const grid_edge& other) const { return !operator==(other); }

  //----------------------------------------------------------------------------
  bool operator<(const grid_edge& other) const {
    for (size_t i = 0; i < n; ++i) {
      if (this->first.iterators[i].i() < other.first.iterators[i].i())
        return true;
      if (this->first.iterators[i].i() > other.first.iterators[i].i())
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
      if (this->second.iterators[i].i() < other.second.iterators[i].i())
        return true;
      if (this->second.iterators[i].i() > other.second.iterators[i].i())
        return false;
    }
    return false;
  }

  //----------------------------------------------------------------------------
  bool operator<=(const grid_edge& other) const {
    return operator==(other) || operator<(other);
  }

  //----------------------------------------------------------------------------
  bool operator>(const grid_edge& other) const {
    for (size_t i = 0; i < n; ++i) {
      if (this->first.iterators[i].i > other.first.iterators[i].i) return true;
      if (this->first.iterators[i].i < other.first.iterators[i].i) return false;
    }
    for (size_t i = 0; i < n; ++i) {
      if (this->second.iterators[i].i > other.second.iterators[i].i)
        return true;
      if (this->second.iterators[i].i < other.second.iterators[i].i)
        return false;
    }
    return false;
  }

  //----------------------------------------------------------------------------
  bool operator>=(const grid_edge& other) const {
    return operator==(other) || operator>(other);
  }

  //----------------------------------------------------------------------------
  auto to_string() {
    return this->first.to_string() + ' ' + this->second.to_string();
  }
};

//==============================================================================

template <size_t n, typename real_t>
inline auto& operator<<(std::ostream& out, const grid_edge<n, real_t>& e) {
  out << e.first << " - " << e.second;
  return out;
}

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
