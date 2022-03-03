#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_SIMPLEX_AT_RETURN_TYPE_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_SIMPLEX_AT_RETURN_TYPE_H
//==============================================================================
namespace tatooine::detail::unstructured_simplicial_grid {
//==============================================================================
template <typename VertexHandle, std::size_t NumVerticesPerSimplex,
          std::size_t I = 0, typename... Ts>
struct simplex_at_return_type_impl {
  using type =
      typename simplex_at_return_type_impl<VertexHandle, NumVerticesPerSimplex,
                                           I + 1, Ts..., VertexHandle>::type;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename VertexHandle, std::size_t NumVerticesPerSimplex,
          typename... Ts>
struct simplex_at_return_type_impl<VertexHandle, NumVerticesPerSimplex,
                                   NumVerticesPerSimplex, Ts...> {
  using type = std::tuple<Ts...>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename VertexHandle, std::size_t NumVerticesPerSimplex>
using simplex_at_return_type =
    typename simplex_at_return_type_impl<VertexHandle,
                                         NumVerticesPerSimplex>::type;
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
#endif
