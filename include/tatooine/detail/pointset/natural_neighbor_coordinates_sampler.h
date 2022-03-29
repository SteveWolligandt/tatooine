#ifdef TATOOINE_HAS_CGAL_SUPPORT
#ifndef TATOOINE_DETAIL_POINTSET_NATURAL_NEIGHBOR_COORDINATES_SAMPLER_H
#define TATOOINE_DETAIL_POINTSET_NATURAL_NEIGHBOR_COORDINATES_SAMPLER_H
//==============================================================================
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T>
struct natural_neighbor_coordinates_sampler
    : field<natural_neighbor_coordinates_sampler<Real, NumDimensions, T>, Real,
            NumDimensions, T> {
  using this_type =
      natural_neighbor_coordinates_sampler<Real, NumDimensions, T>;
  using parent_type = field<this_type, Real, NumDimensions, T>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  using vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<T>;

  using cgal_kernel     = CGAL::Exact_predicates_inexact_constructions_kernel;
  using cgal_coord_type = cgal_kernel::FT;
  using cgal_point      = cgal_kernel::Point_2;
  using triangulation_type = CGAL::Delaunay_triangulation_2<cgal_kernel>;

  //==========================================================================
  pointset_type const&        m_pointset;
  vertex_property_type const& m_property;
  triangulation_type          m_triangulation;
  //==========================================================================
  natural_neighbor_coordinates_sampler(pointset_type const&        ps,
                                       vertex_property_type const& property)
      : m_pointset{ps}, m_property{property} {
    for (auto const v : ps.vertices()) {
      m_triangulation.insert(cgal_point(ps[v].x(), ps[v].y()));
    }
  }
  //--------------------------------------------------------------------------
  natural_neighbor_coordinates_sampler(
      natural_neighbor_coordinates_sampler const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  natural_neighbor_coordinates_sampler(
      natural_neighbor_coordinates_sampler&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(natural_neighbor_coordinates_sampler const&)
      -> natural_neighbor_coordinates_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(natural_neighbor_coordinates_sampler&&) noexcept
      -> natural_neighbor_coordinates_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~natural_neighbor_coordinates_sampler() = default;
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> tensor_type {
    using point_coordinate_vector =
        std::vector<std::pair<cgal_point, cgal_coord_type> >;
    // coordinates computation
    auto p      = cgal_point{x.x(), x.y()};  // query point
    auto coords = point_coordinate_vector{};
    auto result = CGAL::natural_neighbor_coordinates_2(
        m_triangulation, p, std::back_inserter(coords));
    if (!result.third) {
      return parent_type::ood_tensor();
    }
    auto norm = result.second;
    std::cout << "Coordinate computation successful." << std::endl;
    std::cout << "Normalization factor: " << norm << std::endl;
    std::cout << "Coordinates for point: (" << p
              << ") are the following: " << std::endl;
    for (std::size_t i = 0; i < coords.size(); ++i)
      std::cout << "  Point: (" << coords[i].first
                << ") coeff: " << coords[i].second << std::endl;
  }
};
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
#endif
