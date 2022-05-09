#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_VERTEX_PROPERTY_SAMPLER_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_VERTEX_PROPERTY_SAMPLER_H
//==============================================================================
#include <tatooine/unstructured_simplicial_grid.h>
//==============================================================================
namespace tatooine::detail::unstructured_simplicial_grid {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions,
          std::size_t SimplexDim, typename T>
struct vertex_property_sampler
    : field<vertex_property_sampler<Real, NumDimensions, SimplexDim, T>, Real,
            NumDimensions, T> {
  using grid_type =
      tatooine::unstructured_simplicial_grid<Real, NumDimensions, SimplexDim>;
  using this_type = vertex_property_sampler<Real, NumDimensions, SimplexDim, T>;
  using parent_type =
      field<vertex_property_sampler<Real, NumDimensions, SimplexDim, T>, Real,
            NumDimensions, T>;
  using real_type = Real;
  using pos_type  = typename grid_type::pos_type;
  using typed_vertex_property_type =
      typename grid_type::template typed_vertex_property_type<T>;

  static auto constexpr num_dimensions() { return NumDimensions; }

 private:
  grid_type const&                  m_grid;
  typed_vertex_property_type const& m_prop;
  //--------------------------------------------------------------------------
 public:
  vertex_property_sampler(grid_type const&                  grid,
                          typed_vertex_property_type const& prop)
      : m_grid{grid}, m_prop{prop} {}
  //--------------------------------------------------------------------------
  auto grid() const -> auto const& { return m_grid; }
  auto property() const -> auto const& { return m_prop; }
  //--------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> T {
    return evaluate(
        x, std::make_index_sequence<grid_type::num_vertices_per_simplex()>{});
  }
  //--------------------------------------------------------------------------
  template <std::size_t... VertexSeq>
  [[nodiscard]] auto evaluate(pos_type const& x,
                              std::index_sequence<VertexSeq...> /*seq*/) const
      -> T {
    auto simplex_handles = m_grid.hierarchy().nearby_simplices(x);
    if (simplex_handles.empty()) {
      return parent_type::ood_tensor();
    }
    for (auto t : simplex_handles) {
      auto const            vs = m_grid.simplex_at(t);
      static constexpr auto NV = grid_type::num_vertices_per_simplex();
      auto                  A  = mat<Real, NV, NV>::ones();
      auto                  b  = vec<Real, NV>::ones();
      for (std::size_t r = 0; r < num_dimensions(); ++r) {
        (
            [&]() {
              if (VertexSeq > 0) {
                A(r, VertexSeq) = m_grid[std::get<VertexSeq>(vs)](r) -
                                  m_grid[std::get<0>(vs)](r);
              } else {
                ((A(r, VertexSeq) = 0), ...);
              }
            }(),
            ...);

        b(r) = x(r) - m_grid[std::get<0>(vs)](r);
      }
      auto const barycentric_coord = *solve(A, b);
      Real const eps               = 1e-8;
      if (((barycentric_coord(VertexSeq) >= -eps) && ...) &&
          ((barycentric_coord(VertexSeq) <= 1 + eps) && ...)) {
        return (
            (m_prop[std::get<VertexSeq>(vs)] * barycentric_coord(VertexSeq)) +
            ...);
      }
    }
    return parent_type::ood_tensor();
  }
};
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
#endif
