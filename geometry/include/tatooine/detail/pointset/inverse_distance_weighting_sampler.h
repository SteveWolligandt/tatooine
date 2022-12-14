#ifndef TATOOINE_DETAIL_POINTSET_INVERSE_DISTANCE_WEIGHTING_SAMPLER_H
#define TATOOINE_DETAIL_POINTSET_INVERSE_DISTANCE_WEIGHTING_SAMPLER_H
//==============================================================================
namespace tatooine::detail::pointset {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename T>
  requires(flann_available())
struct inverse_distance_weighting_sampler
    : field<inverse_distance_weighting_sampler<Real, NumDimensions, T>, Real,
            NumDimensions, T> {
  using this_type = inverse_distance_weighting_sampler<Real, NumDimensions, T>;
  using parent_type = field<this_type, Real, NumDimensions, T>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  using pointset_t    = tatooine::pointset<Real, NumDimensions>;
  using vertex_handle = typename pointset_t::vertex_handle;
  using vertex_property_type =
      typename pointset_t::template typed_vertex_property_type<T>;
  //==========================================================================
  pointset_t const&        m_pointset;
  vertex_property_type const& m_property;
  Real                     m_radius = 1;
  //==========================================================================
  inverse_distance_weighting_sampler(pointset_t const&        ps,
                                     vertex_property_type const& property,
                                     Real const               radius = 1)
      : m_pointset{ps}, m_property{property}, m_radius{radius} {}
  //--------------------------------------------------------------------------
  inverse_distance_weighting_sampler(
      inverse_distance_weighting_sampler const&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  inverse_distance_weighting_sampler(
      inverse_distance_weighting_sampler&&) noexcept = default;
  //--------------------------------------------------------------------------
  auto operator=(inverse_distance_weighting_sampler const&)
      -> inverse_distance_weighting_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator=(inverse_distance_weighting_sampler&&) noexcept
      -> inverse_distance_weighting_sampler& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ~inverse_distance_weighting_sampler() = default;
  //==========================================================================
  [[nodiscard]] auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> tensor_type {
    auto [indices, squared_distances] =
        m_pointset.nearest_neighbors_radius_raw(x, m_radius);
    if (indices.empty()) {
      return parent_type::ood_tensor();
    }
    auto accumulated_prop_val = T{};
    auto accumulated_weight   = Real{};

    auto index_it = begin(indices);
    auto squared_dist_it  = begin(squared_distances);
    for (; index_it != end(indices); ++index_it, ++squared_dist_it) {
      auto const& property_value = m_property[vertex_handle{*index_it}];
      if (*squared_dist_it == 0) {
        return property_value;
      };
      auto const weight = 1 / *squared_dist_it;
      accumulated_prop_val += property_value * weight;
      accumulated_weight += weight;
    }
    return accumulated_prop_val / accumulated_weight;
  }
};
//==============================================================================
}  // namespace tatooine::detail::pointset
//==============================================================================
#endif
