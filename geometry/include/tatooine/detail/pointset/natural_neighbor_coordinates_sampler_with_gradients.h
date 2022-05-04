#if TATOOINE_CGAL_AVAILABLE
#ifndef TATOOINE_DETAIL_POINTSET_NATURAL_NEIGHBOR_COORDINATES_SAMPLER_WITH_GRADIENTS_H
#define TATOOINE_DETAIL_POINTSET_NATURAL_NEIGHBOR_COORDINATES_SAMPLER_WITH_GRADIENTS_H
//==============================================================================
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/natural_neighbor_coordinates_3.h>
#include <tatooine/cgal.h>
#include <tatooine/concepts.h>
#include <tatooine/tensor_type_traits.h>
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename ValueType,
          typename Gradient>
//requires(tensor_rank<ValueType> + 1 == tensor_rank<Gradient>)
struct natural_neighbor_coordinates_sampler_with_gradients
    : field<natural_neighbor_coordinates_sampler_with_gradients<
                Real, NumDimensions, ValueType, Gradient>,
            Real, NumDimensions, ValueType> {
  using this_type =
      natural_neighbor_coordinates_sampler_with_gradients<Real, NumDimensions,
                                                          ValueType, Gradient>;
  using parent_type = field<this_type, Real, NumDimensions, ValueType>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_type = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_type::vertex_handle;
  using vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<ValueType>;
  using gradient_vertex_property_type =
      typename pointset_type::template typed_vertex_property_type<Gradient>;

  using cgal_kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
  using cgal_triangulation_type =
      cgal::delaunay_triangulation_with_info<NumDimensions, cgal_kernel,
                                             vertex_handle>;
  using cgal_point = cgal_triangulation_type::Point;

  //==========================================================================
  pointset_type const&                 m_p;
  vertex_property_type const&          m_z;
  gradient_vertex_property_type const& m_g;
  cgal_triangulation_type              m_triangulation;
  //==========================================================================
  natural_neighbor_coordinates_sampler_with_gradients(
      pointset_type const& ps, vertex_property_type const& property,
      gradient_vertex_property_type const& gradients_property)
      : m_p{ps}, m_z{property}, m_g{gradients_property} {
    auto points = std::vector<std::pair<cgal_point, vertex_handle>>{};
    points.reserve(ps.vertices().size());
    for (auto v : ps.vertices()) {
      points.emplace_back(cgal_point{ps[v](0), ps[v](1)}, v);
    }

    m_triangulation = cgal_triangulation_type{begin(points), end(points)};
  }
  //--------------------------------------------------------------------------
  natural_neighbor_coordinates_sampler_with_gradients(
      natural_neighbor_coordinates_sampler_with_gradients const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  natural_neighbor_coordinates_sampler_with_gradients(
      natural_neighbor_coordinates_sampler_with_gradients&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(natural_neighbor_coordinates_sampler_with_gradients const&)
      -> natural_neighbor_coordinates_sampler_with_gradients& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(natural_neighbor_coordinates_sampler_with_gradients&&) noexcept
      -> natural_neighbor_coordinates_sampler_with_gradients& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~natural_neighbor_coordinates_sampler_with_gradients() = default;
  //==========================================================================
 private:
  template <std::size_t... Is>
  [[nodiscard]] auto evaluate(pos_type const& x,
                              std::index_sequence<Is...> /*seq*/) const
      -> tensor_type {
    using nnc_per_vertex_type =
        std::vector<std::pair<typename cgal_triangulation_type::Vertex_handle,
                              cgal_kernel::FT>>;

    // coordinates computation
    auto       nnc_per_vertex = nnc_per_vertex_type{};
    auto const result = CGAL::natural_neighbor_coordinates_2(
        m_triangulation, cgal_point{x(Is)...}, std::back_inserter(nnc_per_vertex),
        CGAL::Identity<
            std::pair<typename cgal_triangulation_type::Vertex_handle,
                      cgal_kernel::FT>>{});
    if (!result.third) {
      return parent_type::ood_tensor();
    }
    auto const norm = 1 / result.second;

    auto Z0 = [&] {
      auto sum = ValueType{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const v = cgal_handle->info();
        auto const lambda_i = coeff * norm;
        sum += lambda_i * m_z[v];
      }
      return sum;
    }();

    auto       xi   = [&] {
      auto numerator   = ValueType{};
      auto denominator = real_type{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const v = cgal_handle->info();
        auto const  lambda_i = coeff * norm;
        auto const& p_i      = m_p[v];
        auto const& g_i      = m_g[v];
        auto const  xi_i     = [&] {
          if constexpr (tensor_rank<ValueType> == 0) {
            return m_z[v] + dot(g_i, x - p_i);
          } else {
            return m_z[v] + transposed(g_i) * (x - p_i);
          }
        }();
        auto const  w = lambda_i / euclidean_distance(x, p_i);
        numerator += w * xi_i;
        denominator += w;
      }
      return numerator / denominator;
    }();

    auto       alpha   = [&] {
      auto numerator   = ValueType{};
      auto denominator = real_type{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const v = cgal_handle->info();
        auto const  lambda_i = coeff * norm;
        auto const& p_i      = m_p[v];
        auto const  w        = lambda_i / squared_euclidean_distance(x, p_i);
        numerator += lambda_i;
        denominator += w;
      }
      return numerator / denominator;
    }();

    auto       beta   = [&] {
      auto sum   = real_type{};
      for (auto const& [cgal_handle, coeff] : nnc_per_vertex) {
        auto const v = cgal_handle->info();
        auto const lambda_i = coeff * norm;
        auto const& p_i      = m_p[v];
        sum += lambda_i * squared_euclidean_distance(x, p_i);
      }
      return sum;
    }();

    return (alpha * Z0 + beta * xi) / (alpha + beta);
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
