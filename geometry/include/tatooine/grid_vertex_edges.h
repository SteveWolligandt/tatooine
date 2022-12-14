#ifndef TATOOINE_GRID_VERTEX_EDGES_H
#define TATOOINE_GRID_VERTEX_EDGES_H

//==============================================================================

#include <iostream>
#include "grid_edge.h"
#include "grid_vertex_neighbors.h"

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t N>
struct grid_vertex_edges : grid_vertex_neighbors<Real, N> {
  explicit grid_vertex_edges(const grid_vertex<Real, N>& center)
      : grid_vertex_neighbors<Real, N>(center) {}

  //----------------------------------------------------------------------------

  struct iterator : grid_vertex_neighbors<Real, N>::iterator {
    auto operator*() {
      if (edges()->center == this->v) {
        std::cout << "[ ";
        for (const auto& comp : *edges()->m_begin_vertex) {
          std::cout << comp << ' ';
        }
        std::cout << "] - [";
        for (const auto& comp : *edges()->center) { std::cout << comp << ' '; }
        std::cout << "] - [";
        for (const auto& comp : *edges()->m_end_vertex) {
          std::cout << comp << ' ';
        }
        std::cout << "]\n";
      }
      return grid_edge<Real, N>(edges()->center, this->v);
    }
    auto edges() {
      return reinterpret_cast<grid_vertex_edges*>(this->m_subgrid);
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

  //----------------------------------------------------------------------------
};

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
