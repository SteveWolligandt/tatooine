#ifndef TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
#define TATOOINE_AGRANOVSKY_FLOWMAP_DISCRETIZATION_H
//==============================================================================
#include <tatooine/regular_flowmap_discretization.h>
#include <tatooine/staggered_flowmap_discretization.h>
//==============================================================================
namespace tatooine {
//==============================================================================
// template <floating_point Real, std::size_t NumDimensions>
// struct staggered_flowmap_discretization<
//     regular_flowmap_discretization<Real, NumDimensions>> {
//   using real_type = Real;
//   static auto constexpr num_dimensions() {
//     return NumDimensions;
//   }
//   using vec_type = vec<real_type, num_dimensions()>;
//   using pos_type = vec_type;
//   using internal_flowmap_discretization_type =
//       regular_flowmap_discretization<Real, NumDimensions>;
//   //============================================================================
//   std::list<internal_flowmap_discretization_type> m_steps;
//   std::list<pointset<real_type, num_dimensions()>> m_backward_steps;
//   //============================================================================
//   template <typename Flowmap, typename... InternalFlowmapArgs>
//   staggered_flowmap_discretization(Flowmap&& flowmap, arithmetic auto const t0,
//                                    arithmetic auto const tau,
//                                    arithmetic auto const delta_t,
//                                    InternalFlowmapArgs&&... args) {
//     auto const t1     = t0 + tau;
//     real_type cur_t0 = t0;
//     while (cur_t0 + 1e-10 < t0 + tau) {
//       auto cur_tau = delta_t;
//       if (cur_t0 + cur_tau > t1) {
//         cur_tau = t0 + tau - cur_t0;
//       }
//       auto const& fw_step =
//           m_steps.emplace_back(std::forward<Flowmap>(flowmap), cur_t0, cur_tau,
//                                std::forward<InternalFlowmapArgs>(args)...);
//       cur_t0 += cur_tau;
//       auto& bw_step = m_backward_steps.emplace_back();
//       auto& bw_flowmap = bw_step.vertex_property<pos_type>("flowmap");
//       fw_step.grid(forward).vertices().iterate_indices([&](auto const... is) {
//         auto const v =
//             bw_step.insert_vertex(fw_step.flowmap(forward)(is...));
//         bw_flowmap[v] = fw_step.grid()(is...);
//       });
//   }
//   //----------------------------------------------------------------------------
//   auto steps() const -> auto const& {
//     return m_steps;
//   }
//   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//   auto steps() -> auto& {
//     return m_steps;
//   }
//   //----------------------------------------------------------------------------
//   /// Evaluates flow map in forward direction at time t0 with maximal available
//   /// advection time.
//   /// \param x position
//   /// \returns phi(x, t0, t1 - t0)
//   auto sample(pos_type x, forward_tag const tag) const {
//     for (auto const& step : m_steps) {
//       x = step.sample(x, tag);
//     }
//     return x;
//   }
//   //----------------------------------------------------------------------------
//   /// Evaluates flow map in forward direction at time t0 with maximal available
//   /// advection time.
//   /// \param x position
//   /// \returns phi(x, t1, t0 - t1)
//   auto sample(pos_type x, backward_tag const tag) const {
//     for (auto it = m_steps.rbegin(); it != m_steps.rend(); ++it) {
//       auto const& step = *it;
//       x = step.sample(x, tag);
//     }
//     return x;
//   }
// }
//   //----------------------------------------------------------------------------
//   auto steps() const -> auto const& {
//     return m_steps;
//   }
//   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//   auto steps() -> auto& {
//     return m_steps;
//   }
//   //----------------------------------------------------------------------------
//   /// Evaluates flow map in forward direction at time t0 with maximal available
//   /// advection time.
//   /// \param x position
//   /// \returns phi(x, t0, t1 - t0)
//   auto sample(pos_type x, forward_tag const tag) const {
//     for (auto const& step : m_steps) {
//       x = step.sample(x, tag);
//     }
//     return x;
//   }
//   //----------------------------------------------------------------------------
//   /// Evaluates flow map in forward direction at time t0 with maximal available
//   /// advection time.
//   /// \param x position
//   /// \returns phi(x, t1, t0 - t1)
//   auto sample(pos_type x, backward_tag const tag) const {
//     for (auto it = m_steps.rbegin(); it != m_steps.rend(); ++it) {
//       auto const& step = *it;
//       x = step.sample(x, tag);
//     }
//     return x;
//   }
// };
template <typename Real, std::size_t NumDimensions>
using agranovsky_flowmap_discretization = staggered_flowmap_discretization<
    regular_flowmap_discretization<Real, NumDimensions>>;
template <typename Real>
using AgranovskyFlowmapDiscretization2 =
    agranovsky_flowmap_discretization<Real, 2>;
template <typename Real>
using AgranovskyFlowmapDiscretization3 =
    agranovsky_flowmap_discretization<Real, 3>;
template <std::size_t NumDimensions>
using AgranovskyFlowmapDiscretization =
    agranovsky_flowmap_discretization<real_number, NumDimensions>;
using agranovsky_flowmap_discretization2 =
    AgranovskyFlowmapDiscretization<2>;
using agranovsky_flowmap_discretization3 =
    AgranovskyFlowmapDiscretization<3>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
