#if TATOOINE_CGAL_AVAILABLE || defined(TATOOINE_DOC_ONLY)
//==============================================================================
#ifndef TATOOINE_CGAL_NATURAL_NEIGHBOR_COORDINATES_H
#define TATOOINE_CGAL_NATURAL_NEIGHBOR_COORDINATES_H
//==============================================================================
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/natural_neighbor_coordinates_3.h>
#include <tatooine/cgal/delaunay_triangulation.h>

#include <vector>
//==============================================================================
namespace tatooine::cgal {
/// \defgroup cgal CGAL Wrappers
/// \brief Templated Wrappers for CGAL types.
//==============================================================================
template <std::size_t NumDimensions, typename Traits,
          typename TriangulationDataStructure>
requires(NumDimensions == 2)
auto natural_neighbor_coordinates(
    delaunay_triangulation<NumDimensions, Traits,
                           TriangulationDataStructure> const &triangulation,
    typename delaunay_triangulation<NumDimensions, Traits,
                                    TriangulationDataStructure>::Point const
        &query) {
  auto nnc = std::vector<std::pair<
      typename delaunay_triangulation<
          NumDimensions, Traits, TriangulationDataStructure>::Vertex_handle,
      typename Traits::FT>>{};
  return std::pair{
      CGAL::natural_neighbor_coordinates_2(
          triangulation, query, std::back_inserter(nnc),
          CGAL::Identity<typename std::decay_t<decltype(nnc)>::value_type>{}),
      std::move(nnc)};
}
//==============================================================================
template <std::size_t NumDimensions, typename Traits,
          typename TriangulationDataStructure>
requires(NumDimensions == 3)
auto natural_neighbor_coordinates(
    delaunay_triangulation<NumDimensions, Traits,
                           TriangulationDataStructure> const &triangulation,
    typename delaunay_triangulation<NumDimensions, Traits,
                                    TriangulationDataStructure>::Point const
        &query) {
  auto nnc               = std::vector<std::pair<
      typename delaunay_triangulation<
          NumDimensions, Traits, TriangulationDataStructure>::Vertex_handle,
      typename Traits::FT>>{};
  auto norm_coeff_sibson = typename Traits::FT{};
  return std::pair{
      CGAL::sibson_natural_neighbor_coordinates_3(
          triangulation, query, std::back_inserter(nnc), norm_coeff_sibson),
      std::move(nnc)};
}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
