#ifndef TATOOINE_STAGGERED_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_STAGGERED_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalFlowmapDiscretization>
struct staggered_flowmap_discretization {
  using internal_flowmap_discretization_type = InternalFlowmapDiscretization;
  using real_type = typename internal_flowmap_discretization_type::real_type;
  static auto constexpr num_dimensions() -> std::size_t {
    return internal_flowmap_discretization_type::num_dimensions();
  }
  using vec_type = vec<real_type, num_dimensions()>;
  using pos_type = vec_type;
  //============================================================================
  std::vector<internal_flowmap_discretization_type> m_steps = {};
  //============================================================================
  template <typename Flowmap, typename... InternalFlowmapArgs>
  staggered_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
                                   arithmetic auto const tau,
                                   arithmetic auto const delta_t,
                                   InternalFlowmapArgs&&... args) {
    auto       cur_t0 = real_type(t0);
    auto const t_end = static_cast<real_type>(t0) + static_cast<real_type>(tau);
    m_steps.reserve(static_cast<std::size_t>((t_end - t0) / delta_t) + 2);
    static auto const eps = real_type(1e-10);
    while (cur_t0 + eps < t0 + tau) {
      auto cur_tau = static_cast<real_type>(delta_t);
      if (cur_t0 + cur_tau > t_end) {
        cur_tau = static_cast<real_type>(t0) + static_cast<real_type>(tau) -
                  static_cast<real_type>(cur_t0);
      }
      m_steps.emplace_back(std::forward<Flowmap>(flowmap), cur_t0, cur_tau,
                           std::forward<InternalFlowmapArgs>(args)...);
      cur_t0 += cur_tau;
    }
  }
  //============================================================================
  auto num_steps() const { return m_steps.size(); }
  //----------------------------------------------------------------------------
  auto steps() const -> auto const& {
    return m_steps;
  }
  //----------------------------------------------------------------------------
  auto steps() -> auto& { return m_steps; }
  //============================================================================
  auto step(std::size_t const i) const -> auto const& { return m_steps[i]; }
  //----------------------------------------------------------------------------
  auto step(std::size_t const i) -> auto& { return m_steps[i]; }
  //============================================================================
  /// Evaluates flow map in forward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t_end - t0)
  auto sample(pos_type x, forward_tag const tag) const {
    for (auto const& step : steps()) {
      x = step.sample(x, tag);
    }
    return x;
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in backward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t_end, t0 - t_end)
  auto sample(pos_type x, backward_tag const tag) const {
    for (auto step = steps().rbegin(); step != steps().rend(); ++step) {
      x = step->sample(x, tag);
    }
    return x;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
