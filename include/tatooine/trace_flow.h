#ifndef TATOOINE_TRACE_FLOW_H
#define TATOOINE_TRACE_FLOW_H
//==============================================================================
#include <tatooine/line.h>
#include <tatooine/ode/vclibs/rungekutta43.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename V, typename Real, std::size_t N, typename T0Real,
          typename TauReal>
auto trace_flow(vectorfield<V, Real, N> const& v, vec<Real, N> const& x0,
                T0Real const t0, TauReal const tau) {
  auto  solver   = ode::vclibs::rungekutta43<Real, N> {};
  auto  l        = line<Real, N>{};
  auto& param    = l.parameterization();
  auto& tangents = l.tangents();

  solver.solve(v, x0, t0, tau, [&](auto const& x, auto const t, auto const dx) {
    if (tau > 0) {
      l.push_back(x);
      param.back()    = t;
      tangents.back() = dx;
    } else {
      l.push_front(x);
      param.front()    = t;
      tangents.front() = dx;
    }
  });
  return l;
}
//==============================================================================
}
//==============================================================================
#endif
