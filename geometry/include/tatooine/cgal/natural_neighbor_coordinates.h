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
template <std::size_t NumDimensions, typename Traits, typename Info>
requires(NumDimensions == 2)
auto natural_neighbor_coordinates(
    delaunay_triangulation_with_info<NumDimensions, Traits, Info> const
                                                                 &triangulation,
    typename delaunay_triangulation_with_info<NumDimensions, Traits,
                                              Info>::Point const &query,
    std::vector<std::pair<typename delaunay_triangulation_with_info<
                              NumDimensions, Traits, Info>::Vertex_handle,
                          typename Traits::FT>>                  &nnc) {
  return CGAL::natural_neighbor_coordinates_2(
      triangulation, query, std::back_inserter(nnc),
      CGAL::Identity<typename std::decay_t<decltype(nnc)>::value_type>{});
}
//==============================================================================
template <std::size_t NumDimensions, typename Traits, typename Info>
requires(NumDimensions == 3)
auto natural_neighbor_coordinates(
    delaunay_triangulation_with_info<NumDimensions, Traits, Info> const
        &triangulation,
    typename delaunay_triangulation_with_info<
        NumDimensions, Traits, Info>::Point const &query,
    std::vector<
        std::pair<typename delaunay_triangulation_with_info<
                      NumDimensions, Traits, Info>::Vertex_handle,
                  typename Traits::FT>> &nnc) {
  auto norm_coeff_sibson = real_number{};
  return CGAL::sibson_natural_neighbor_coordinates_3(
      triangulation, query, std::back_inserter(nnc), norm_coeff_sibson);
}
//==============================================================================
}  // namespace tatooine::cgal
//==============================================================================
#endif
//==============================================================================
#endif
