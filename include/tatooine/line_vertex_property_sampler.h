#ifndef TATOOINE_LINE_VERTEX_PROPERTY_SAMPLER_H
#define TATOOINE_LINE_VERTEX_PROPERTY_SAMPLER_H
//==============================================================================
#include <tatooine/line.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N, typename Prop,
          template <typename> typename InterpolationKernel>
struct line_vertex_property_sampler {
  using line_t   = line<Real, N>;
  using handle_t = typename line<Real, N>::vertex_handle;
  // using vertex_property_t = typename line_t::template vertex_property_t<T>;
  using parameterization_property_t =
      typename line_t::parameterization_property_t;

 private:
  line_t const&                      m_line;
  Prop const&                        m_prop;
  parameterization_property_t const& m_parameterization;

 public:
  line_vertex_property_sampler(line_t const&            line,
                               Prop const& prop)
      : m_line{line},
        m_prop{prop},
        m_parameterization{m_line.parameterization()} {}

  auto operator()(Real t) const{
    auto range =
        std::pair{m_line.vertices().front(), m_line.vertices().back()};
    while (range.second.i - range.first.i > 1) {
      auto const center = handle_t{(range.first.i + range.second.i)/2};
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
template <typename Real, size_t N, typename Prop>
struct line_vertex_property_sampler<Real, N, Prop, interpolation::cubic> {
  using line_t   = line<Real, N>;
  using handle_t = typename line<Real, N>::vertex_handle;
  using tangent_property_t =
      typename line_t::tangent_property_t;
  using parameterization_property_t =
      typename line_t::parameterization_property_t;
  using value_type = std::decay_t<decltype(std::declval<Prop>()[std::declval<handle_t>()])>;

 private:
  line_t const&                                 m_line;
  Prop const&                                   m_prop;
  parameterization_property_t const&            m_parameterization;
  std::vector<interpolation::cubic<value_type>> m_interpolants;

 public:
  line_vertex_property_sampler(line_t const& line, Prop const& prop)
      : m_line{line},
        m_prop{prop},
        m_parameterization{m_line.parameterization()} {
    size_t const stencil_size = 3;
    auto const                               half         = stencil_size / 2;
    std::vector<value_type> derivatives;

    auto const derivative = [&](auto const v) {
      auto       lv         = half > v.i ? handle_t{0} : v - half;
      auto const rv         = lv.i + stencil_size - 1 >= m_line.num_vertices()
                                  ? handle_t{m_line.num_vertices() - 1}
                                  : lv + stencil_size - 1;
      auto const rpotential = stencil_size - (rv.i - lv.i + 1);
      lv = rpotential > lv.i ? handle_t{0} : lv - rpotential;

      std::vector<real_t> ts(stencil_size);
      size_t              i = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        ts[i] = m_parameterization[vi] - m_parameterization[v];
      }
      auto coeffs      = finite_differences_coefficients(1, ts);
      auto derivative  = value_type{};
      i                = 0;
      for (auto vi = lv; vi <= rv; ++vi, ++i) {
        derivative += m_prop[vi] * coeffs[i];
      }
      return derivative;
    };
    auto dfdt0 = derivative(handle_t{0});
    for (size_t i = 0; i < m_line.num_vertices() - 1; ++i) {
      auto const dfdt1 = derivative(handle_t{i + 1});
      auto const dy =
          m_parameterization[handle_t{i + 1}] - m_parameterization[handle_t{i}];
      m_interpolants.emplace_back(m_prop[handle_t{i}], m_prop[handle_t{i + 1}],
                                  dfdt0 * dy, dfdt1 * dy);
      dfdt0 = dfdt1;
    }
  }
  //----------------------------------------------------------------------------
  auto operator()(Real t) const{
    auto range =
        std::pair{m_line.vertices().front(), m_line.vertices().back()};
    while (range.second.i - range.first.i > 1) {
      auto const center = handle_t{(range.first.i + range.second.i)/2};
      if (t < m_parameterization[center]) {
        range.second = center;
      } else {
        range.first = center;
      }
    }
    auto const lt = m_parameterization[range.first];
    auto const rt = m_parameterization[range.second];
    t             = (t - lt) / (rt - lt);

    return m_interpolants[range.first.i](t);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
