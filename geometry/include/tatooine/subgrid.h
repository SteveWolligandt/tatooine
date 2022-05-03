#ifndef TATOOINE_SUBGRID_H
#define TATOOINE_SUBGRID_H

#include <cstddef>
#include <functional>

#include "grid_vertex.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
struct subgrid {
  subgrid(const grid_vertex<Real, N>& begin, const grid_vertex<Real, N>& end)
      : m_begin_vertex(begin), m_end_vertex(end) {}

  //------------------------------------------------------------------------

  grid_vertex<Real, N> m_begin_vertex, m_end_vertex;

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

template <typename Real, size_t N>
struct subgrid<Real, N>::vertex_iterator {
  grid_vertex<Real, N> v;
  subgrid<Real, N>*    m_subgrid;
  //----------------------------------------------------------------------------
  auto operator++() -> auto& {
    ++v.iterators.front();
    for (size_t i = 0; i < N - 1; ++i) {
      if (v.iterators[i] == m_subgrid->m_end_vertex.iterators[i]) {
        v.iterators[i] = m_subgrid->m_begin_vertex.iterators[i];
        ++v.iterators[i + 1];
      }
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator--() -> auto& {
    for (size_t i = 0; i < N; ++i) {
      if (v.iterators[i] == m_subgrid->begin_vertex.iterators[i]) {
        v.iterators[i] = m_subgrid->end_vertex.iterators[i];
        --v.iterators[i];
      } else {
        --v.iterators[i];
        break;
      }
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  auto operator*() { return v; }
  auto operator==(const vertex_iterator& other) { return v == other.v; }
  auto operator!=(const vertex_iterator& other) { return v != other.v; }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
