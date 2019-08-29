#ifndef GRID_SUBGRID_H
#define GRID_SUBGRID_H

#include <cstddef>
#include <functional>
#include "grid_vertex.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <size_t n, typename real_t>
struct subgrid {
  subgrid(const grid_vertex<n, real_t>& begin,
          const grid_vertex<n, real_t>& end)
      : m_begin_vertex(begin), m_end_vertex(end) {}

  //------------------------------------------------------------------------

  grid_vertex<n, real_t> m_begin_vertex, m_end_vertex;

  //------------------------------------------------------------------------

  struct vertex_iterator;

  //----------------------------------------------------------------------------

  auto begin() { return vertex_iterator{m_begin_vertex, this}; }

  //----------------------------------------------------------------------------

  auto end() {
    auto actual_end_vertex             = m_begin_vertex;
    actual_end_vertex.iterators.back() = m_end_vertex.iterators.back();
    return vertex_iterator{actual_end_vertex, this};
  }

};

//==============================================================================

template <size_t n, typename real_t>
struct subgrid<n, real_t>::vertex_iterator {
  grid_vertex<n, real_t> v;
  subgrid<n, real_t>*    m_subgrid;

  //----------------------------------------------------------------------------

  auto& operator++() {
    ++v.iterators.front();
    for (size_t i = 0; i < n - 1; ++i)
      if (v.iterators[i] == m_subgrid->m_end_vertex.iterators[i]) {
        v.iterators[i] = m_subgrid->m_begin_vertex.iterators[i];
        ++v.iterators[i + 1];
      }
    return *this;
  }

  //----------------------------------------------------------------------------

  auto& operator--() {
    for (size_t i = 0; i < n; ++i)
      if (v.iterators[i] == m_subgrid->begin_vertex.iterators[i]) {
        v.iterators[i] = m_subgrid->end_vertex.iterators[i];
        --v.iterators[i];
      } else {
        --v.iterators[i];
        break;
      }
    return *this;
  }

  //----------------------------------------------------------------------------

  auto operator*() { return v; }
  bool operator==(const vertex_iterator& other) { return v == other.v; }
  bool operator!=(const vertex_iterator& other) { return v != other.v; }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
