#ifndef TATOOINE_DETAIL_LINE_VERTEX_PROPERTY_SAMPLER_H
#define TATOOINE_DETAIL_LINE_VERTEX_PROPERTY_SAMPLER_H
//==============================================================================
#include <tatooine/line.h>
//==============================================================================
namespace tatooine::detail::line {
//==============================================================================
template <typename Real, std::size_t NumDimensions, typename Prop,
          template <typename> typename InterpolationKernel>
struct vertex_property_sampler {
  using line_type   = tatooine::line<Real, NumDimensions>;
  using handle_type = typename line_type::vertex_handle;
  using parameterization_property_type =
      typename line_type::parameterization_property_type;

 private:
  line_type const&                      m_line;
  Prop const&                           m_prop;
  parameterization_property_type const& m_parameterization;

 public:
  vertex_property_sampler(line_type const& line, Prop const& prop)
      : m_line{line},
        m_prop{prop},
        m_parameterization{m_line.parameterization()} {}

  auto operator()(Real t) const {
    auto range = std::pair{m_line.vertices().front(), m_line.vertices().back()};
    while (range.second.index() - range.first.index() > 1) {
      auto const center =
          handle_type{(range.first.index() + range.second.index()) / 2};
      if (t < m_parameterization[center]) {
        range.second = center;
      } else {
        range.first = center;
      }
    }
    auto const lt = m_parameterization[range.first];
    auto const rt = m_parameterization[range.second];
    t             = (t - lt) / (rt - lt);

    return InterpolationKernel{m_prop[range.first], m_prop[range.second]}(t);
  }
};
template <typename Real, std::size_t NumDimensions, typename Prop>
struct vertex_property_sampler<Real, NumDimensions, Prop,
                               interpolation::cubic> {
  using line_type   = tatooine::line<Real, NumDimensions>;
  using handle_type = typename line<Real, NumDimensions>::vertex_handle;
  using tangent_property_type = typename line_type::tangent_property_type;
  using parameterization_property_type =
      typename line_type::parameterization_property_type;
  using value_type =
      std::decay_t<decltype(std::declval<Prop>()[std::declval<handle_type>()])>;

 private:
  line_type const&                              m_line;
  Prop const&                                   m_prop;
  parameterization_property_type const&         m_parameterization;
  std::vector<interpolation::cubic<value_type>> m_interpolants;

 public:
  vertex_property_sampler(line_type const& line, Prop const& prop)
      : m_line{line},
        m_prop{prop},
        m_parameterization{m_line.parameterization()} {
    std::size_t const       stencil_size = 3;
    auto const              half         = stencil_size / 2;
    std::vector<value_type> derivatives;

    auto const derivative = [&](auto const v) {
      auto       lv = half > v.index() ? handle_type{0} : v - half;
      auto const rv = lv.index() + stencil_size - 1 >= m_line.vertices().size()
                          ? handle_type{m_line.vertices().size() - 1}
                          : lv + stencil_size - 1;
      auto const rpotential = stencil_size - (rv.index() - lv.index() + 1);
      lv = rpotential > lv.index() ? handle_type{0} : lv - rpotential;

      std::vector<real_t> ts(stencil_size);
      std::size_t         i = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        ts[i] = m_parameterization[vi] - m_parameterization[v];
      }
      auto coeffs     = finite_differences_coefficients(1, ts);
      auto derivative = value_type{};
      i               = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        derivative += m_prop[vi] * coeffs[i];
      }
      return derivative;
    };
    auto dfdt0 = derivative(handle_type{0});
    for (std::size_t i = 0; i < m_line.vertices().size() - 1; ++i) {
      auto const dfdt1 = derivative(handle_type{i + 1});
      auto const dy    = m_parameterization[handle_type{i + 1}] -
                      m_parameterization[handle_type{i}];
      m_interpolants.emplace_back(m_prop[handle_type{i}],
                                  m_prop[handle_type{i + 1}], dfdt0 * dy,
                                  dfdt1 * dy);
      dfdt0 = dfdt1;
    }
  }
  //----------------------------------------------------------------------------
  auto operator()(Real t) const {
    auto range = std::pair{m_line.vertices().front(), m_line.vertices().back()};
    while (range.second.index() - range.first.index() > 1) {
      auto const center =
          handle_type{(range.first.index() + range.second.index()) / 2};
      if (t < m_parameterization[center]) {
        range.second = center;
      } else {
        range.first = center;
      }
    }
    auto const lt = m_parameterization[range.first];
    auto const rt = m_parameterization[range.second];
    t             = (t - lt) / (rt - lt);

    return m_interpolants[range.first.index()](t);
  }
};
//==============================================================================
}  // namespace tatooine::detail::line
//==============================================================================
#endif
