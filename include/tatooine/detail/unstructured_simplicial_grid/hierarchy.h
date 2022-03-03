#ifndef TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_HIERARCHY_H
#define TATOOINE_DETAIL_UNSTRUCTURED_SIMPLICIAL_GRID_HIERARCHY_H
//==============================================================================
#include <tatooine/uniform_tree_hierarchy.h>
//==============================================================================
namespace tatooine::detail::unstructured_simplicial_grid {
//==============================================================================
template <typename Mesh, floating_point Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
struct hierarchy_impl {
  using type = int;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, floating_point Real, std::size_t NumDimensions,
          std::size_t SimplexDim>
using hierarchy =
    typename hierarchy_impl<Mesh, Real, NumDimensions, SimplexDim>::type;
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, floating_point Real>
struct hierarchy_impl<Mesh, Real, 3, 3> {
  using type = uniform_tree_hierarchy<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, floating_point Real>
struct hierarchy_impl<Mesh, Real, 2, 2> {
  using type = uniform_tree_hierarchy<Mesh>;
};
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
template <typename Mesh, floating_point Real>
struct hierarchy_impl<Mesh, Real, 3, 2> {
  using type = uniform_tree_hierarchy<Mesh>;
};
//==============================================================================
}  // namespace tatooine::detail::unstructured_simplicial_grid
//==============================================================================
#endif
