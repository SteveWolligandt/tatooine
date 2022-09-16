#ifndef TATOOINE_DETAIL_RECTILINEAR_GRID_INVERSE_DISTANCE_WEIGHTING_VERTEX_PROPERTY_SAMPLER_H
#define TATOOINE_DETAIL_RECTILINEAR_GRID_INVERSE_DISTANCE_WEIGHTING_VERTEX_PROPERTY_SAMPLER_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/pointset.h>
//==============================================================================
namespace tatooine::detail::rectilinear_grid {
//==============================================================================
template <typename Grid, typename Property>
struct inverse_distance_weighting_vertex_property_sampler
    : tatooine::field<
          inverse_distance_weighting_vertex_property_sampler<Grid, Property>,
          typename Grid::real_type, Grid::num_dimensions(),
          typename Property::value_type> {
  static auto constexpr num_dimensions() -> std::size_t {
    return Grid::num_dimensions();
  }

  using this_type =
      inverse_distance_weighting_vertex_property_sampler<Grid, Property>;
  using grid_type     = Grid;
  using property_type = Property;
  using real_type     = typename Grid::real_type;
  using value_type    = typename Property::value_type;
  using parent_type =
      tatooine::field<this_type, real_type, num_dimensions(), value_type>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  using pointset_type =
      tatooine::pointset<typename Grid::real_type, Grid::num_dimensions()>;
  //----------------------------------------------------------------------------
  Grid const&     m_grid;
  Property const& m_property;
  pointset_type   m_points;
  real_type       m_radius;
  //----------------------------------------------------------------------------
  inverse_distance_weighting_vertex_property_sampler(Grid const&     g,
                                                     Property const& p,
                                                     real_type const radius)
      : m_grid{g}, m_property{p}, m_radius{radius} {
    m_points.vertices().reserve(m_grid.vertices().size());
    m_grid.vertices().iterate_indices([this](auto const... is) {
      m_points.insert_vertex(m_grid.vertex_at(is...));
    });
  }
  //----------------------------------------------------------------------------
  auto evaluate(pos_type const& x, real_type const /*t*/) const -> tensor_type {
    auto [indices, squared_distances] =
        m_points.nearest_neighbors_radius_raw(x, m_radius);
    if (indices.empty()) {
      return parent_type::ood_tensor();
    }
    auto accumulated_prop_val = tensor_type{};
    auto accumulated_weight   = real_type{};

    auto index_it        = begin(indices);
    auto squared_dist_it = begin(squared_distances);
    for (; index_it != end(indices); ++index_it, ++squared_dist_it) {
      auto const& property_value = m_property.plain_at(*index_it);
      if (*squared_dist_it == 0) {
        return property_value;
      };
      auto const dist   = std::sqrt(*squared_dist_it);
      auto const weight = 1 / (dist * dist * dist);
      accumulated_prop_val += property_value * weight;
      accumulated_weight += weight;
    }
    return accumulated_prop_val / accumulated_weight;
  }
};
//==============================================================================
}  // namespace tatooine::detail::rectilinear_grid
//==============================================================================
#endif
