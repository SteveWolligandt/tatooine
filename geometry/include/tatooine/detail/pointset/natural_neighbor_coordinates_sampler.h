#ifdef TATOOINE_HAS_CGAL_SUPPORT
#ifndef TATOOINE_DETAIL_POINTSET_NATURAL_NEIGHBOR_COORDINATES_SAMPLER_H
#define TATOOINE_DETAIL_POINTSET_NATURAL_NEIGHBOR_COORDINATES_SAMPLER_H
//==============================================================================
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/natural_neighbor_coordinates_3.h>
#include <tatooine/cgal.h>
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

  using cgal_kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using triangulation_type =
      cgal::delaunay_triangulation_with_info<NumDimensions, cgal_kernel,
                                             vertex_handle>;
  using cgal_point = typename triangulation_type::Point;

  //==========================================================================
  pointset_type const&        m_pointset;
  vertex_property_type const& m_property;
  triangulation_type          m_triangulation;
  //==========================================================================
  natural_neighbor_coordinates_sampler(pointset_type const&        ps,
                                       vertex_property_type const& property)
      : m_pointset{ps}, m_property{property} {
    auto points = std::vector<std::pair<cgal_point, vertex_handle>>{};
    points.reserve(ps.vertices().size());
    for (auto v : ps.vertices()) {
      points.emplace_back(cgal_point{ps[v](0), ps[v](1)}, v);
    }

    m_triangulation = triangulation_type{begin(points), end(points)};
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
 private:
  template <std::size_t... Is>
  [[nodiscard]] auto evaluate(pos_type const& x,
                              std::index_sequence<Is...> /*seq*/) const
      -> tensor_type {
    using point_coordinate_vector =
        std::vector<std::pair<typename triangulation_type::Vertex_handle, cgal_kernel::FT>>;

    // coordinates computation
    auto       coords = point_coordinate_vector{};
    auto const p      = cgal_point{x(Is)...};  // query point
    auto const result = CGAL::natural_neighbor_coordinates_2(
        m_triangulation, p, std::back_inserter(coords),
        CGAL::Identity<std::pair<typename triangulation_type::Vertex_handle,
                                 cgal_kernel::FT>>{});
    if (!result.third) {
      return parent_type::ood_tensor();
    }
    auto const norm = 1 / result.second;
    // std::cout << "Coordinate computation successful." << std::endl;
    // std::cout << "Normalization factor: " << norm << std::endl;
    // std::cout << "Coordinates for point: (" << p
    //           << ") are the following: " << std::endl;
    auto t = tensor_type{};
    for (auto const& [handle, coeff] : coords) {
      t += m_property[handle->info()] * coeff * norm;
    }
    return t;
  }
  //----------------------------------------------------------------------------
 public:
  [[nodiscard]] auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> tensor_type {
    return evaluate(x, std::make_index_sequence<NumDimensions>{});
  }
};
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
#endif
