#ifndef TATOOINE_GRID_VERTEX_HANDLE_H
#define TATOOINE_GRID_VERTEX_HANDLE_H
//==============================================================================
#include <array>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <size_t NumDimensions>
struct grid_vertex_handle {
 private:
  std::array<size_t, NumDimensions> m_indices;
  size_t m_plain_index;

 public:
  static constexpr auto num_dimensions() { return NumDimensions; }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if_integral<Is...> = true>
#endif
  constexpr grid_vertex_handle(Is const... is)
      : m_indices{static_cast<size_t>(is)...} {
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral Int>
#else
  template <typename Int, enable_if_integral<Int> = true>
#endif
  constexpr grid_vertex_handle(std::array<Int, num_dimensions()> const& is,
                               size_t const plain_index)
      : m_indices{begin(is), end(is)}, m_plain_index{plain_index} {
  }
  //----------------------------------------------------------------------------
  constexpr grid_vertex_handle(std::array<size_t, num_dimensions()> const& is,
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
  constexpr auto operator==(grid_vertex_handle const& other) const {
    return m_plain_index == other.m_plain_index;
  }
  //----------------------------------------------------------------------------
  constexpr auto operator!=(grid_vertex_handle const& other) const {
    return m_plain_index != other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<(grid_vertex_handle const& other) const -> bool {
    return m_plain_index < other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator<=(grid_vertex_handle const& other) const -> bool {
    return m_plain_index <= other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>(grid_vertex_handle const& other) const -> bool {
    return m_plain_index > other.m_plain_index;
  }
  //--------------------------------------------------------------------------
  constexpr auto operator>=(grid_vertex_handle const& other) const -> bool {
    return m_plain_index >= other.m_plain_index;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/rectilinear_grid.h>
#endif
