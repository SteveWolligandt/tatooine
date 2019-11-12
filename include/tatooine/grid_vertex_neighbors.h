#ifndef TATOOINE_GRID_VERTEX_NEIGHBORS_H
#define TATOOINE_GRID_VERTEX_NEIGHBORS_H

#include <cstddef>
#include "grid_vertex.h"
#include "subgrid.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
struct grid_vertex_neighbors : subgrid<Real, N> {
  grid_vertex<Real, N> center;

  //----------------------------------------------------------------------------
  grid_vertex_neighbors(const grid_vertex<Real, N>& c)
      : subgrid<Real, N>{c, c}, center(c) {
    for (size_t i = 0; i < N; ++i) {
      if (this->m_begin_vertex.iterators[i] !=
          this->m_begin_vertex.iterators[i].begin()) {
        --this->m_begin_vertex.iterators[i];
      }
      for (size_t j = 0; j < 2; ++j) {
        if (this->m_end_vertex.iterators[i] !=
            this->m_end_vertex.iterators[i].end()) {
          ++this->m_end_vertex.iterators[i];
        }
      }
    }
  }

  //----------------------------------------------------------------------------
  struct iterator : subgrid<Real, N>::vertex_iterator {
    auto& operator++() {
      subgrid<Real, N>::vertex_iterator::operator++();
      if (this->v == neighbors()->center) {
        subgrid<Real, N>::vertex_iterator::operator++();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    auto& operator--() {
      subgrid<Real, N>::vertex_iterator::operator--();
      if (this->v == neighbors()->center) {
        subgrid<Real, N>::vertex_iterator::operator--();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    auto neighbors() {
      return reinterpret_cast<grid_vertex_neighbors*>(this->m_subgrid);
    }
  };

  //----------------------------------------------------------------------------
  auto begin() {
    iterator it{this->m_begin_vertex, this};
    if ((this->m_begin_vertex == this->center)) { ++it; }
    return it;
  }

  //----------------------------------------------------------------------------
  auto end() {
    auto actual_end_vertex             = this->m_begin_vertex;
    actual_end_vertex.iterators.back() = this->m_end_vertex.iterators.back();
    return iterator{actual_end_vertex, this};
  }
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
