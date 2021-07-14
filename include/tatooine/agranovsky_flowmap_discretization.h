#ifndef TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/field.h>
#include <tatooine/grid.h>
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
    while (cur_t0 + delta_t < t0 + tau) {
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
  auto steps() const -> auto const& {
    return m_stepped_flowmap_discretizations;
  }
  auto steps() -> auto& { return m_stepped_flowmap_discretizations; }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
