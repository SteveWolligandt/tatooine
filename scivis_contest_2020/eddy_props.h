#ifndef TATOOINE_SCIVIS_CONTEST_2020_EDDY_PROPS
#define TATOOINE_SCIVIS_CONTEST_2020_EDDY_PROPS
//==============================================================================
#include <tatooine/concepts.h>

#include <tuple>

#include "integrate_pathline.h"
//==============================================================================
namespace tatooine::scivis_contest_2020 {
//==============================================================================
template <typename V>
auto eddy_props(V const& v, typename V::pos_t const& x,
                real_number auto const t) {
  auto const Jf             = diff(v, 1e-8);
  auto const eulerian_J     = Jf(x, t);
  auto const eulerian_S     = (eulerian_J + transposed(eulerian_J)) / 2;
  auto const eulerian_Omega = (eulerian_J - transposed(eulerian_J)) / 2;
  auto const eulerian_Q = (sqr_norm(eulerian_Omega) - sqr_norm(eulerian_S)) / 2;

  auto pathline        = integrate_pathline(v, x, t);
  auto pathline_Q_prop = pathline.template add_vertex_property<double>("Q");

  // for each vertex of the pathline calculate properties
  for (size_t i = 0; i < pathline.num_vertices(); ++i) {
    typename decltype(pathline)::vertex_idx vi{i};
    auto const& lagrangian_x    = pathline.vertex_at(i);
    auto const& lagrangian_t    = pathline.parameterization_at(i);
    auto const  lagrangian_J    = Jf(lagrangian_x, lagrangian_t);
    auto const  lagrangian_S    = (lagrangian_J + transposed(lagrangian_J)) / 2;
    auto const lagrangian_Omega = (lagrangian_J - transposed(lagrangian_J)) / 2;
    // auto const SS    = S * S;
    // auto const OO    = Omega * Omega;
    // auto const SSOO  = SS + OO;

    // vec const vort{J(2, 1) - J(1, 2),
    //                J(0, 2) - J(2, 0),
    //                J(1, 0) - J(0, 1)};
    // vorticity
    // pathline_prop[vi] = length(vort);
    // Q
    pathline_Q_prop[vi] =
        (sqr_norm(lagrangian_Omega) - sqr_norm(lagrangian_S)) / 2;
    // lambda2
    // pathline_prop[vi] = eigenvalues_sym(SSOO)(1);
  }
  auto const lagrangian_Q = pathline.integrate_property(pathline_Q_prop);
  auto       Q_time       = [&](double const threshold) {
    double Q_time = 0;
    for (size_t i = 0; i < pathline.num_vertices() - 1; ++i) {
      typename decltype(pathline)::vertex_idx vi{i};
      typename decltype(pathline)::vertex_idx vj{i + 1};
      auto const t0 = pathline.parameterization_at(i);
      auto const t1 = pathline.parameterization_at(i + 1);
      auto const Q0 = pathline_Q_prop[vi];
      auto const Q1 = pathline_Q_prop[vj];
      if (Q0 >= threshold && Q1 >= threshold) {
        Q_time += t1 - t0;
      } else if (Q0 >= threshold && Q1 < threshold) {
        auto const t_root =
            ((t1 - t0) * threshold - Q0 * t1 + Q1 * t0) / (Q1 - Q0);
        Q_time += t_root - t0;
      } else if (Q0 < threshold && Q1 >= threshold) {
        auto const t_root =
            ((t1 - t0) * threshold - Q0 * t1 + Q1 * t0) / (Q1 - Q0);
        Q_time += t1 - t_root;
      }
    }
    return Q_time;
  };
  return std::tuple{eulerian_Q, lagrangian_Q, Q_time(0), Q_time(0.05), Q_time(0.1), pathline};
}
//==============================================================================
}  // namespace tatooine::scivis_contest_2020
//==============================================================================
#endif
