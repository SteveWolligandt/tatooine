#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_SIMPLEX_CONTAINER_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_SIMPLEX_CONTAINER_H
//==============================================================================
#include <tatooine/unstructured_simplicial_grid.h>
//==============================================================================
namespace tatooine::detail::unstructured_simplicial_grid {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
struct simplex_container {
  using grid_type =
      tatooine::unstructured_simplicial_grid<Real, NumDimensions, SimplexDim>;
  using handle_type = typename grid_type::simplex_handle;
  //----------------------------------------------------------------------------
  struct iterator : iterator_facade<iterator> {
    struct sentinel_type {};
    iterator() = default;
    iterator(handle_type const ch, grid_type const* ps) : m_ch{ch}, m_ps{ps} {}
    iterator(iterator const& other) : m_ch{other.m_ch}, m_ps{other.m_ps} {}

   private:
    handle_type      m_ch{};
    grid_type const* m_ps = nullptr;

   public:
    constexpr auto increment() {
      do {
        ++m_ch;
      } while (!m_ps->is_valid(m_ch));
    }
    constexpr auto decrement() {
      do {
        --m_ch;
      } while (!m_ps->is_valid(m_ch));
    }

    [[nodiscard]] constexpr auto equal(iterator const& other) const {
      return m_ch == other.m_ch;
    }
    [[nodiscard]] auto dereference() const { return m_ch; }

    constexpr auto at_end() const {
      return m_ch.index() == m_ps->simplex_index_data().size();
    }
  };
  //--------------------------------------------------------------------------
  grid_type const* m_grid;
  //--------------------------------------------------------------------------
  auto begin() const {
    iterator vi{handle_type{0}, m_grid};
    if (!m_grid->is_valid(*vi)) {
      ++vi;
    }
    return vi;
  }
  //--------------------------------------------------------------------------
  auto end() const { return iterator{handle_type{size()}, m_grid}; }
  //--------------------------------------------------------------------------
  auto size() const {
    return m_grid->simplex_index_data().size() /
               m_grid->num_vertices_per_simplex() -
           m_grid->invalid_simplices().size();
  }
  auto data_container() const -> auto const& {
    return m_grid->simplex_index_data();
  }
  auto data() const { return m_grid->simplex_index_data().data(); }
  auto operator[](std::size_t const i) const { return m_grid->at(handle_type{i}); }
  auto operator[](std::size_t const i) { return m_grid->at(handle_type{i}); }
  auto operator[](handle_type const i) const { return m_grid->at(i); }
  auto operator[](handle_type const i) { return m_grid->at(i); }
  auto at(std::size_t const i) const { return m_grid->at(handle_type{i}); }
  auto at(std::size_t const i) { return m_grid->at(handle_type{i}); }
  auto at(handle_type const i) const { return m_grid->at(i); }
  auto at(handle_type const i) { return m_grid->at(i); }
};
//------------------------------------------------------------------------------
template <floating_point Real, size_t NumDimensions, std::size_t SimplexDim>
auto begin(simplex_container<Real, NumDimensions, SimplexDim> simplices) {
  return simplices.begin();
}
//------------------------------------------------------------------------------
template <floating_point Real, size_t NumDimensions, std::size_t SimplexDim>
auto end(simplex_container<Real, NumDimensions, SimplexDim> simplices) {
  return simplices.end();
}
//------------------------------------------------------------------------------
template <floating_point Real, size_t NumDimensions, std::size_t SimplexDim>
auto size(simplex_container<Real, NumDimensions, SimplexDim> simplices) {
  return simplices.size();
}
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
template <tatooine::floating_point Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
inline constexpr bool std::ranges::enable_borrowed_range<
    typename tatooine::detail::unstructured_simplicial_grid::simplex_container<
        Real, NumDimensions, SimplexDim>> = true;
//==============================================================================
#endif
