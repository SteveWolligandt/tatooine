#ifndef TATOOINE_ANALYTICAL_NUMERICAL_MODIFIED_DOUBLEGYRE_H
#define TATOOINE_ANALYTICAL_NUMERICAL_MODIFIED_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/algorithm.h>
#include <tatooine/numerical_flowmap.h>
#include <tatooine/field.h>
#include <tatooine/line.h>
#include <tatooine/linspace.h>
#include <tatooine/type_traits.h>

#include <algorithm>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
template <typename Real>
struct modified_doublegyre : vectorfield<modified_doublegyre<Real>, Real, 2> {
  using this_type   = modified_doublegyre<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  static constexpr Real pi      = M_PI;
  static constexpr Real epsilon = 0.25;
  static constexpr Real omega   = 2 * pi * 0.1;
  static constexpr Real A       = 0.1;
  static constexpr Real c       = -0.2040811331;
  static constexpr Real cc      = c * c;
  static constexpr Real d       = 9.964223388;
  //============================================================================
  constexpr auto evaluate(pos_type const& x, Real const t) const
      -> tensor_type {
    Real const a  = epsilon * gcem::sin(omega * (t + timeoffset(t)));
    Real const b  = 1 - 2 * a;
    Real const f  = a * x(0) * x(0) + b * x(0);
    Real const df = 2 * a * x(0) + b;

    return tensor_type{-pi * A * gcem::sin(pi * f) * gcem::cos(pi * x(1)),
                    pi * A * gcem::cos(pi * f) * gcem::sin(pi * x(1)) * df};
  }
  //----------------------------------------------------------------------------
  constexpr static auto timeoffset(Real const t) {
    Real const r = pi / 5 * t + d;

    auto const q =
        std::clamp<Real>((4 * pi * c * sin(r) - 4 * gcem::asin(2 * c * cos(r))) /
                        (pi * (1 - cc * sin(r) * sin(r))),
                    -1, 1);

    Real const p = 5 / pi * gcem::asin(q) - t;
    return p;
  }
  //----------------------------------------------------------------------------
  struct hyperbolic_trajectory_type {
    auto at(Real const t) const {
      return vec<Real, 2>{c * gcem::sin(pi / 5 * t + d) + 1, 0};
    }
    auto operator()(Real const t) const { return at(t); }
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto hyperbolic_trajectory() const {
    return hyperbolic_trajectory_type{};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto hyperbolic_trajectory(Real t) const {
    return hyperbolic_trajectory_type{}(t);
  }
  //----------------------------------------------------------------------------
  struct hyperbolic_trajectory_spacetime_type {
    auto at(Real const t) const {
      return vec<Real, 3>{c * gcem::sin(pi / 5 * t + d) + 1, 0, t};
    }
    auto operator()(Real const t) const { return at(t); }
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto hyperbolic_trajectory_spacetime() const {
    return hyperbolic_trajectory_spacetime_type{};
  }
  //============================================================================
  template <template <typename, std::size_t> typename ODESolver>
  struct lagrangian_coherent_structure_type {
    Real m_t0;
    Real m_eps;
    this_type const& m_v;
    numerical_flowmap<this_type const&, ODESolver> m_flowmap;
    //--------------------------------------------------------------------------
    lagrangian_coherent_structure_type(this_type const& v, Real const t0,
                                       Real const eps)
        : m_t0{t0}, m_eps{eps}, m_v{v}, m_flowmap{flowmap(v)} {}
    //--------------------------------------------------------------------------
    auto at(Real const t) const {
      return m_flowmap(m_v.hyperbolic_trajectory(t) + Vec2<Real>{0, m_eps}, t,
                       m_t0 - t);
      //return m_flowmap(m_v.hyperbolic_trajectory(t) + Vec2<Real>{0, 1-m_eps}, t-m_t0,
      //                 t);
    }
    auto operator()(Real const t) const { return at(t); }
  };
  //----------------------------------------------------------------------------
  template <template <typename, std::size_t>
            typename ODESolver = ode::boost::rungekuttafehlberg78>
  auto lagrangian_coherent_structure(Real const t,
                                     Real const eps = 1e-10) const {
    return lagrangian_coherent_structure_type<ODESolver>{*this, t, eps};
  }
};
//==============================================================================
modified_doublegyre()->modified_doublegyre<real_number>;
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
