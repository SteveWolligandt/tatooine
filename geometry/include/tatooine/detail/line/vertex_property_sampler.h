#ifndef TATOOINE_DETAIL_LINE_VERTEX_PROPERTY_SAMPLER_H
#define TATOOINE_DETAIL_LINE_VERTEX_PROPERTY_SAMPLER_H
//==============================================================================
#include <tatooine/line.h>
#include <tatooine/nan.h>
//==============================================================================
namespace tatooine::detail::line {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename Property,
          template <typename> typename InterpolationKernel>
struct vertex_property_sampler {
  using line_type     = tatooine::line<Real, NumDimensions>;
  using handle_type   = typename line_type::vertex_handle;
  using property_type = Property;
  using parameterization_property_type =
      typename line_type::parameterization_property_type;

 private:
  line_type const&                      m_line;
  property_type const&                  m_property;
  parameterization_property_type const& m_parameterization;

 public:
  vertex_property_sampler(line_type const& line, property_type const& property)
      : m_line{line},
        m_property{property},
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

    auto const interpolant = InterpolationKernel{m_property[range.first], m_property[range.second]};
    return interpolant(t);
  }
};
//==============================================================================
template <floating_point Real, std::size_t NumDimensions, typename Property>
struct vertex_property_sampler<Real, NumDimensions, Property,
                               interpolation::cubic> {
  using real_type = Real;
  using line_type             = tatooine::line<Real, NumDimensions>;
  using handle_type           = typename line_type::vertex_handle;
  using tangent_property_type = typename line_type::tangent_property_type;
  using parameterization_property_type =
      typename line_type::parameterization_property_type;
  using property_type = Property;
  using value_type    = typename property_type::value_type;

 private:
  line_type const&                              m_line;
  property_type const&                          m_property;
  parameterization_property_type const&         m_parameterization;
  std::vector<interpolation::cubic<value_type>> m_interpolants;

 public:
  vertex_property_sampler(line_type const& line, property_type const& property)
      : m_line{line},
        m_property{property},
        m_parameterization{m_line.parameterization()} {
    if (m_line.vertices().size() < 2) {
      return;
    }
    auto const stencil_size =
        min(std::size_t(3), m_line.vertices().size());
    auto const half         = stencil_size / 2;
    auto       derivatives  = std::vector<value_type>{};

    auto const derivative = [&](handle_type const v) -> value_type {
      auto       lv = half > v.index() ? handle_type{0} : v - half;
      auto const rv = lv.index() + stencil_size - 1 >= m_line.vertices().size()
                          ? handle_type{m_line.vertices().size() - 1}
                          : lv + stencil_size - 1;
      auto const rpotential = stencil_size - (rv.index() - lv.index() + 1);
      lv = rpotential > lv.index() ? handle_type{0} : lv - rpotential;

      auto ts = std::vector<Real>(stencil_size);
      auto i  = std::size_t{};
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        ts[i] = m_parameterization[vi] - m_parameterization[v];
      }
      auto coeffs     = finite_differences_coefficients(1, ts);
      auto derivative = value_type{};
      i               = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        derivative += m_property[vi] *
                      static_cast<tatooine::value_type<value_type>>(coeffs[i]);
      }
      return derivative;
    };
    auto dfdt0 = derivative(handle_type{0});
    for (std::size_t i = 0; i < m_line.vertices().size() - 1; ++i) {
      auto const dfdt1 = derivative(handle_type{i + 1});
      auto const dy    = m_parameterization[handle_type{i + 1}] -
                      m_parameterization[handle_type{i}];
      m_interpolants.emplace_back(
          m_property[handle_type{i}], m_property[handle_type{i + 1}],
          value_type(dfdt0 * dy), value_type(dfdt1 * dy));
      dfdt0 = dfdt1;
    }
  }
  //----------------------------------------------------------------------------
  auto operator()(Real t) const -> value_type{
    if (m_line.vertices().size() < 2) {
      if constexpr (tensor_rank<value_type> == 0) {
        return nan<real_type>();
      } else {
        return value_type::fill(nan<real_type>());
      }
    }
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
