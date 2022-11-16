#if TATOOINE_CGAL_AVAILABLE
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
  using cgal_triangulation_type = std::conditional_t<
      NumDimensions == 2,
      cgal::delaunay_triangulation_with_info<2, vertex_handle, cgal_kernel>,
      std::conditional_t<
          NumDimensions == 3,
          cgal::delaunay_triangulation_with_info<
              3, vertex_handle, cgal_kernel,
              cgal::delaunay_triangulation_simplex_base_with_circumcenter<
                  3, cgal_kernel>>,
          void>>;
  using cgal_point = typename cgal_triangulation_type::Point;
  //==========================================================================
private:
  pointset_type const &m_pointset;
  vertex_property_type const &m_property;
  cgal_triangulation_type m_triangulation;
  //==========================================================================
public:
  natural_neighbor_coordinates_sampler(pointset_type const &ps,
                                       vertex_property_type const &property)
      : m_pointset{ps}, m_property{property} {
    auto points = std::vector<std::pair<cgal_point, vertex_handle>>{};
    points.reserve(ps.vertices().size());
    for (auto v : ps.vertices()) {
      [&]<std::size_t... Is>(std::index_sequence<Is...> /*seq*/){
        points.emplace_back(cgal_point{ps[v](Is)...}, v);
      }(std::make_index_sequence<NumDimensions>{});
    }
    m_triangulation = cgal_triangulation_type{begin(points), end(points)};
  }
  //--------------------------------------------------------------------------
  natural_neighbor_coordinates_sampler(
      natural_neighbor_coordinates_sampler const &) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  natural_neighbor_coordinates_sampler(
      natural_neighbor_coordinates_sampler &&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(natural_neighbor_coordinates_sampler const &)
      -> natural_neighbor_coordinates_sampler & = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(natural_neighbor_coordinates_sampler &&) noexcept
      -> natural_neighbor_coordinates_sampler & = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~natural_neighbor_coordinates_sampler() = default;
  //==========================================================================
private:
  template <std::size_t... Is>
  [[nodiscard]] auto evaluate(pos_type const &x,
                              std::index_sequence<Is...> /*seq*/) const
      -> tensor_type {
    // coordinates computation
    auto const [result, coords] = cgal::natural_neighbor_coordinates<
        NumDimensions, typename cgal_triangulation_type::Geom_traits,
        typename cgal_triangulation_type::Triangulation_data_structure>(
        m_triangulation, cgal_point{x(Is)...});
    if (!result.third) {
      return parent_type::ood_tensor();
    }
    auto const norm = 1 / result.second;
    auto t = tensor_type{};
    for (auto const &[handle, coeff] : coords) {
      t += m_property[handle->info()] * coeff * norm;
    }
    return t;
  }
  //----------------------------------------------------------------------------
public:
  [[nodiscard]] auto evaluate(pos_type const &x, real_type const /*t*/) const
      -> tensor_type {
    return evaluate(x, std::make_index_sequence<NumDimensions>{});
  }
};
//==============================================================================
} // namespace tatooine::detail::pointset
//==============================================================================
#endif
#endif
