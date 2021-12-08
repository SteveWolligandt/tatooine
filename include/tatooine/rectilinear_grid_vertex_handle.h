#ifndef TATOOINE_RECTILINEAR_GRID_VERTEX_HANDLE_H
#define TATOOINE_RECTILINEAR_GRID_VERTEX_HANDLE_H
//==============================================================================
#include <array>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t NumDimensions>
struct rectilinear_grid_vertex_handle {
 private:
  std::array<size_t, NumDimensions> m_indices;
  size_t m_plain_index;

 public:
  static constexpr auto num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
  constexpr rectilinear_grid_vertex_handle(integral auto const... is)
      : m_indices{static_cast<size_t>(is)...} {
  }
  //----------------------------------------------------------------------------
  template <integral Int>
  constexpr rectilinear_grid_vertex_handle(std::array<Int, num_dimensions()> const& is,
                               size_t const plain_index)
      : m_indices{begin(is), end(is)}, m_plain_index{plain_index} {
  }
  //----------------------------------------------------------------------------
  constexpr rectilinear_grid_vertex_handle(std::array<size_t, num_dimensions()> const& is,
                               size_t const plain_index)
      : m_indices{is}, m_plain_index{plain_index} {}
  //----------------------------------------------------------------------------
  constexpr auto indices() const -> auto const& { return m_indices; }
  constexpr auto indices() -> auto& { return m_indices; }
  //----------------------------------------------------------------------------
  constexpr auto index(size_t const i) const -> auto const& { return m_indices[i]; }
  constexpr auto index(size_t const i) -> auto& { return m_indices[i]; }
  //----------------------------------------------------------------------------
  constexpr auto plain_index() const -> auto const& { return m_plain_index; }
  constexpr auto plain_index() -> auto& { return m_plain_index; }
  //----------------------------------------------------------------------------
  constexpr auto operator==(rectilinear_grid_vertex_handle const& other) const {
    return m_plain_index == other.m_plain_index;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator!=(rectilinear_grid_vertex_handle const& other) const {
    return m_plain_index != other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<(rectilinear_grid_vertex_handle const& other) const -> bool {
    return m_plain_index < other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<=(rectilinear_grid_vertex_handle const& other) const -> bool {
    return m_plain_index <= other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>(rectilinear_grid_vertex_handle const& other) const -> bool {
    return m_plain_index > other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>=(rectilinear_grid_vertex_handle const& other) const -> bool {
    return m_plain_index >= other.m_plain_index;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#endif
