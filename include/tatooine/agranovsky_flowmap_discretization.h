#ifndef TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/rectilinear_grid.h>
#include <tatooine/naive_flowmap_discretization.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t N>
struct agranovsky_flowmap_discretization {
  using vec_t = vec<Real, N>;
  using pos_t = vec_t;
  //============================================================================
  std::vector<naive_flowmap_discretization<Real, N>> m_stepped_flowmap_discretizations;
  //============================================================================
  template <typename Flowmap>
  agranovsky_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
                                    arithmetic auto const tau,
                                    arithmetic auto const delta_t,
                                    pos_t const& min, pos_t const& max,
                                    integral auto const... resolution) {
    auto cur_t0 = t0;
    while (cur_t0 < t0 + tau) {
      auto cur_tau = delta_t;
      if (cur_t0 + cur_tau > t0 + tau) {
        cur_tau = t0 + tau - cur_t0;
      }
      m_stepped_flowmap_discretizations.emplace_back(
          std::forward<Flowmap>(flowmap), cur_t0, cur_tau, min, max,
          resolution...);
      cur_t0 += delta_t;
    }
  }
  //----------------------------------------------------------------------------
  auto stepped_flowmap_discretizations() const -> auto const& {
    return m_stepped_flowmap_discretizations;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto stepped_flowmap_discretizations() -> auto& {
    return m_stepped_flowmap_discretizations;
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in forward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t1 - t0)
  auto sample_forward(pos_t x) const {
    for (auto const& step : m_stepped_flowmap_discretizations) {
      x = step.sample_forward(x);
    }
    return x;
  }
  //----------------------------------------------------------------------------
  /// Evaluates flow map in forward direction at time t0 with maximal available
  /// advection time.
  /// \param x position
  /// \returns phi(x, t0, t1 - t0)
  auto sample_backward(pos_t x) const {
    for (auto it = m_stepped_flowmap_discretizations.rbegin();
         it != m_stepped_flowmap_discretizations.rend(); ++it) {
      auto const& step = *it;
      x = step.sample_backward(x);
    }
    return x;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
