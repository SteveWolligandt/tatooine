#ifndef TATOOINE_EDGE_H
#define TATOOINE_EDGE_H
//==============================================================================
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "for_loop.h"
#include "grid_vertex.h"
#include "linspace.h"
#include "math.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
class grid;
template <typename Real, size_t N>
class grid_edge_container;
template <typename Real, size_t N>
class grid_edge_iterator;
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
template <typename Real, size_t N>
class grid_edge_iterator {
  template <size_t... Is>
  static constexpr size_t calc_max_num_edges_per_cell(
      std::index_sequence<Is...> /*seq*/) {
    return (ipow(3, Is) + ...);
  }

  static constexpr size_t calc_max_num_edges_per_cell() {
    return calc_max_num_edges_per_cell(std::make_index_sequence<N>{});
  }

  template <size_t... Is>
  static constexpr auto calc_edge_dirs(std::index_sequence<Is...> /*seq*/) {
    auto   dirs = make_array<vec<int, N>, max_num_edges_per_cell>();
    size_t i    = 0;
    for_loop(
        [&](auto... is) {
          if (((is != 1) || ...)) {
            dirs[i] = vec<int, N>{(2 - int(is) - 1)...};
            for (size_t j = 0; j < N / 2; ++j) {
              tat_swap(dirs[i](j), dirs[i][N - j - 1]);
            }
            ++i;
            return true;
          }
          return false;
        },
        ((void)Is, 3)...);
    return dirs;
  }
  static constexpr auto calc_edge_dirs() {
    return calc_edge_dirs(std::make_index_sequence<N>{});
  }

  static constexpr auto calc_bases() {
    auto bs = make_array<vec<size_t, N>, max_num_edges_per_cell>(
        vec<size_t, N>::zeros());
    auto base_it = begin(bs);
    for (const auto& dir : edge_dirs) {
      for (size_t i = 0; i < N; ++i) {
        if (dir(i) < 0) { base_it->at(i) = 1; }
      }
      ++base_it;
    }
    return bs;
  }

 public:
  static constexpr auto max_num_edges_per_cell = calc_max_num_edges_per_cell();
  static constexpr auto edge_dirs              = calc_edge_dirs();
  static constexpr auto bases                  = calc_bases();

 private:
  grid<Real, N>const*                m_grid;
  grid_vertex_iterator<Real, N> m_vit;
  size_t                        m_edge_idx;

 public:
  explicit grid_edge_iterator(grid<Real, N>const*                  g,
                              grid_vertex_iterator<Real, N>&& vit,
                              size_t                          edge_idx)
      : m_grid{g}, m_vit{std::move(vit)}, m_edge_idx{edge_idx} {}
  auto operator++() -> auto& {
    do {
      ++m_edge_idx;
      if (m_edge_idx == max_num_edges_per_cell) {
        m_edge_idx = 0;
        ++m_vit;
      }
    } while (!is_valid());
    return *this;
  }
  auto operator==(const grid_edge_iterator<Real, N>& other) const -> bool {
    return m_grid == other.m_grid && m_vit == other.m_vit &&
           m_edge_idx == other.m_edge_idx;
  }
  auto operator!=(const grid_edge_iterator<Real, N>& other) const -> bool {
    return m_grid != other.m_grid || m_vit != other.m_vit ||
           m_edge_idx != other.m_edge_idx;
  }
  auto operator<(const grid_edge_iterator<Real, N>& other) const -> bool{
    if (m_vit < other.m_vit) { return true; }
    if (m_vit > other.m_vit) { return false; }
    return m_edge_idx < other.m_edge_idx;
  }
  auto operator*() const -> grid_edge<Real, N> {
    auto v0 = *m_vit;
    for (size_t i = 0; i < N; ++i) { v0.index(i) += bases[m_edge_idx](i); }
    auto v1 = v0;
    for (size_t i = 0; i < N; ++i) { v1.index(i) += edge_dirs[m_edge_idx](i); }
    return grid_edge{std::move(v0), std::move(v1)};
  }
  auto is_valid() const -> bool {
    auto v0     = *m_vit;
    bool is_end = true;
    for (size_t i = 0; i < N; ++i) {
      if (i < N - 1) {
        if (v0.index(i) != 0) {
          is_end = false;
          break;
        }
      } else {
        if (v0.index(i) != m_grid->dimension(i).size()) {
          is_end = false;
          break;
        }
      }
    }
    if (is_end && m_edge_idx == 0) { return true; }

    for (size_t i = 0; i < N; ++i) { v0.index(i) += bases[m_edge_idx](i); }
    auto v1 = v0;
    for (size_t i = 0; i < N; ++i) { v1.index(i) += edge_dirs[m_edge_idx](i); }

    for (size_t i = 0; i < N; ++i) {
      if (v0.index(i) >= m_grid->dimension(i).size()) { return false; }
    }
    for (size_t i = 0; i < N; ++i) {
      if (v1.index(i) >= m_grid->dimension(i).size()) { return false; }
    }
    return true;
  }
};
//==============================================================================
template <typename Real, size_t N>
class grid_edge_container {
  grid<Real, N>const* m_grid;

 public:
  explicit grid_edge_container(grid<Real, N>const* g) : m_grid{g} {}
  auto begin() const {
    return grid_edge_iterator{m_grid, m_grid->vertex_begin(), 0};
  }
  auto end() const {
    return grid_edge_iterator{m_grid, m_grid->vertex_end(), 0};
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
