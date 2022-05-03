#ifndef TATOOINE_STAGGERED_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_STAGGERED_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename InternalFlowmapDiscretization>
struct staggered_flowmap_discretization {
  using real_type = typename InternalFlowmapDiscretization::real_type;
  static auto constexpr num_dimensions() {
    return InternalFlowmapDiscretization::num_dimensions();
  }
  using vec_t = vec<real_type, num_dimensions()>;
  using pos_type = vec_t;
  //============================================================================
  std::list<InternalFlowmapDiscretization> m_steps;
  //============================================================================
  template <typename Flowmap, typename... InternalFlowmapArgs>
  staggered_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
                                   arithmetic auto const tau,
                                   arithmetic auto const delta_t,
                                   InternalFlowmapArgs&&... args) {
    auto const t1     = t0 + tau;
    real_type cur_t0 = t0;
    while (std::abs(cur_t0 - (t0 + tau)) > 1e-10) {
      auto cur_tau = delta_t;
      if (cur_t0 + cur_tau > t1) {
        cur_tau = t0 + tau - cur_t0;
      }
      m_steps.emplace_back(std::forward<Flowmap>(flowmap), cur_t0, cur_tau,
                           std::forward<InternalFlowmapArgs>(args)...);
      cur_t0 += delta_t;
    }
  }
  //----------------------------------------------------------------------------
  auto steps() const -> auto const& {
    return m_steps;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto steps() -> auto& {
    return m_steps;
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in forward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t1 - t0)
  auto sample(pos_type x, forward_tag const tag) const {
    for (auto const& step : m_steps) {
      x = step.sample(x, tag);
    }
    return x;
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in forward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t1, t0 - t1)
  auto sample(pos_type x, backward_tag const tag) const {
    for (auto it = m_steps.rbegin(); it != m_steps.rend(); ++it) {
      auto const& step = *it;
      x = step.sample(x, tag);
    }
    return x;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
