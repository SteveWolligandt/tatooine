#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_SIMPLEX_BASE_H
#define TATOOINE_CGAL_SIMPLEX_BASE_H
//==============================================================================
#include <tatooine/cgal/triangulation_ds_vertex_base.h>
//==============================================================================
namespace tatooine::cgal {
//==============================================================================
/// \defgroup cgal_triangulation_simplex_base Triangulation Face or Cell Base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename Traits, typename SimplexBase>
struct triangulation_simplex_base_impl;
//------------------------------------------------------------------------------
template <typename Traits, typename FaceBase>
struct triangulation_simplex_base_impl<2, Traits, FaceBase> {
  using type = CGAL::Triangulation_face_base_2<Traits, FaceBase>;
};
//------------------------------------------------------------------------------
template <typename Traits, typename CellBase>
struct triangulation_simplex_base_impl<3, Traits, CellBase> {
  using type = CGAL::Triangulation_cell_base_3<Traits, CellBase>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Traits,
          typename SimplexBase = triangulation_ds_simplex_base<NumDimensions>>
using triangulation_simplex_base =
    typename triangulation_simplex_base_impl<NumDimensions, Traits,
                                             SimplexBase>::type;
/// \}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
