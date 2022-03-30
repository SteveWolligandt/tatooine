#if defined(TATOOINE_HAS_CGAL_SUPPORT) || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_H
#define TATOOINE_CGAL_H
//==============================================================================
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
//==============================================================================
namespace tatooine::cgal {
/// \defgroup cgal CGAL Wrappers
/// \brief Templated Wrappers for CGAL types.
//==============================================================================
/// \defgroup cgal_triangulation_ds_vertex_base Triangulation DS Vertex Base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions>
struct triangulation_ds_vertex_base_impl;
//------------------------------------------------------------------------------
template <>
struct triangulation_ds_vertex_base_impl<2> {
  using type = CGAL::Triangulation_ds_vertex_base_2<>;
};
//------------------------------------------------------------------------------
template <>
struct triangulation_ds_vertex_base_impl<3> {
  using type = CGAL::Triangulation_ds_vertex_base_3<>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions>
using triangulation_ds_vertex_base =
    typename triangulation_ds_vertex_base_impl<NumDimensions>::type;
/// \}
//==============================================================================
/// \defgroup cgal_triangulation_ds_face_base Triangulation DS face base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions>
struct triangulation_ds_face_base_impl;
//------------------------------------------------------------------------------
template <>
struct triangulation_ds_face_base_impl<2> {
  using type = CGAL::Triangulation_ds_face_base_2<>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions>
using triangulation_ds_face_base =
    typename triangulation_ds_face_base_impl<NumDimensions>::type;
/// \}
//==============================================================================
/// \defgroup cgal_triangulation_ds_cell_base Triangulation DS cell base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions>
struct triangulation_ds_cell_base_impl;
//------------------------------------------------------------------------------
template <>
struct triangulation_ds_cell_base_impl<3> {
  using type = CGAL::Triangulation_ds_cell_base_3<>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions>
using triangulation_ds_cell_base =
    typename triangulation_ds_cell_base_impl<NumDimensions>::type;
/// \}
//==============================================================================
/// \defgroup cgal_triangulation_data_structure Triangulation data structure
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename VertexBase, typename FaceBase>
struct triangulation_data_structure_impl;
//------------------------------------------------------------------------------
template <typename VertexBase, typename FaceBase>
struct triangulation_data_structure_impl<2, VertexBase, FaceBase> {
  using type = CGAL::Triangulation_data_structure_2<VertexBase, FaceBase>;
};
//------------------------------------------------------------------------------
template <typename VertexBase, typename FaceBase>
struct triangulation_data_structure_impl<3, VertexBase, FaceBase> {
  using type = CGAL::Triangulation_data_structure_3<VertexBase, FaceBase>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions,
          typename VertexBase = triangulation_ds_vertex_base<NumDimensions>,
          typename FaceBase   = triangulation_ds_face_base<NumDimensions>>
using triangulation_data_structure =
    typename triangulation_data_structure_impl<NumDimensions, VertexBase,
                                               FaceBase>::type;
/// \}
//==============================================================================
/// \defgroup cgal_triangulation_vertex_base Triangulation vertex base
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename Traits, typename Vb>
struct triangulation_vertex_base_impl;
//------------------------------------------------------------------------------
template <typename Traits, typename Vb>
struct triangulation_vertex_base_impl<2, Traits, Vb> {
  using type = CGAL::Triangulation_vertex_base_2<Traits, Vb>;
};
//------------------------------------------------------------------------------
template <typename Traits, typename Vb>
struct triangulation_vertex_base_impl<3, Traits, Vb> {
  using type = CGAL::Triangulation_vertex_base_3<Traits, Vb>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Traits,
          typename Vb = triangulation_ds_vertex_base<NumDimensions>>
using triangulation_vertex_base =
    typename triangulation_vertex_base_impl<NumDimensions, Traits, Vb>::type;
/// \}
//==============================================================================
/// \defgroup cgal_triangulation_vertex_base_with_info Triangulation vertex base with info
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename Info, typename Traits, typename Vb>
struct triangulation_vertex_base_with_info_impl;
//------------------------------------------------------------------------------
template <typename Info, typename Traits, typename Vb>
struct triangulation_vertex_base_with_info_impl<2, Info, Traits, Vb> {
  using type = CGAL::Triangulation_vertex_base_with_info_2<Info, Traits, Vb>;
};
//------------------------------------------------------------------------------
template <typename Info, typename Traits, typename Vb>
struct triangulation_vertex_base_with_info_impl<3, Info, Traits, Vb> {
  using type = CGAL::Triangulation_vertex_base_with_info_3<Info, Traits, Vb>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Info, typename Traits,
          typename Vb = triangulation_vertex_base<NumDimensions, Traits>>
using triangulation_vertex_base_with_info =
    typename triangulation_vertex_base_with_info_impl<NumDimensions, Info,
                                                      Traits, Vb>::type;
/// \}
//==============================================================================
/// \defgroup cgal_delaunay_triangulation Delaunay Triangulation
/// \ingroup cgal
/// \{
template <std::size_t NumDimensions, typename Traits, typename Tds>
struct delaunay_triangulation_impl;
//------------------------------------------------------------------------------
template <typename Traits, typename Tds>
struct delaunay_triangulation_impl<2, Traits, Tds> {
  using type = CGAL::Delaunay_triangulation_2<Traits, Tds>;
};
//------------------------------------------------------------------------------
template <typename Traits, typename Tds>
struct delaunay_triangulation_impl<3, Traits, Tds> {
  using type = CGAL::Delaunay_triangulation_3<Traits, Tds>;
};
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Traits, typename Tds>
using delaunay_triangulation =
    typename delaunay_triangulation_impl<NumDimensions, Traits, Tds>::type;
//------------------------------------------------------------------------------
template <std::size_t NumDimensions, typename Traits, typename Info>
using delaunay_triangulation_with_info = typename delaunay_triangulation_impl<
    NumDimensions, Traits,
    triangulation_data_structure<
        NumDimensions, triangulation_vertex_base_with_info<NumDimensions, Info,
                                                           Traits>>>::type;
/// \}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
