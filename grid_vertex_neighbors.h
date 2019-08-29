#ifndef TATOOINE_GRID_VERTEX_NEIGHBORS_H
#define TATOOINE_GRID_VERTEX_NEIGHBORS_H

#include <cstddef>
#include "grid_vertex.h"
#include "subgrid.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <size_t n, typename real_t>
struct grid_vertex_neighbors : subgrid<n, real_t> {
  grid_vertex<n, real_t> center;

  //----------------------------------------------------------------------------
  grid_vertex_neighbors(const grid_vertex<n, real_t>& c)
      : subgrid<n, real_t>{c, c}, center(c) {
    for (size_t i = 0; i < n; ++i) {
      if (this->m_begin_vertex.iterators[i] !=
          this->m_begin_vertex.iterators[i].get().begin()) {
        --this->m_begin_vertex.iterators[i];
      }
      for (size_t j = 0; j < 2; ++j) {
        if (this->m_end_vertex.iterators[i] !=
            this->m_end_vertex.iterators[i].get().end()) {
          ++this->m_end_vertex.iterators[i];
        }
      }
    }
  }

  //----------------------------------------------------------------------------
  struct iterator : subgrid<n, real_t>::vertex_iterator {
    auto& operator++() {
      subgrid<n, real_t>::vertex_iterator::operator++();
      if (this->v == neighbors()->center) {
        subgrid<n, real_t>::vertex_iterator::operator++();
      }
      return *this;
    }

    //--------------------------------------------------------------------------
    auto& operator--() {
      subgrid<n, real_t>::vertex_iterator::operator--();
      if (this->v == neighbors()->center) {
        subgrid<n, real_t>::vertex_iterator::operator--();
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
