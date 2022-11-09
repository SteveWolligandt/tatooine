#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_TRIANGULATION_DS_SIMPLEX_BASE_H
#define TATOOINE_CGAL_TRIANGULATION_DS_SIMPLEX_BASE_H
//==============================================================================
#include <CGAL/Triangulation_ds_cell_base_3.h>
#include <CGAL/Triangulation_ds_face_base_2.h>
//==============================================================================
namespace tatooine::cgal {
//==============================================================================
/// \defgroup cgal_triangulation_ds_simplex_base Triangulation data structure
/// face base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename TDS>
struct triangulation_ds_simplex_base_impl;
//------------------------------------------------------------------------------
template <typename TDS>
struct triangulation_ds_simplex_base_impl<2, TDS> {
  using type = CGAL::Triangulation_ds_face_base_2<TDS>;
};
//------------------------------------------------------------------------------
template <typename TDS>
struct triangulation_ds_simplex_base_impl<3, TDS> {
  using type = CGAL::Triangulation_ds_cell_base_3<TDS>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename TDS = void>
using triangulation_ds_simplex_base =
    typename triangulation_ds_simplex_base_impl<NumDimensions, void>::type;
/// \}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
