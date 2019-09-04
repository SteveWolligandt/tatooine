#ifndef TATOOINE_NEWDOUBLEGYRE_H
#define TATOOINE_NEWDOUBLEGYRE_H

#include <algorithm>
#include "line.h"
#include "linspace.h"
#include "field.h"
#include "symbolic_field.h"

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================
template <typename Real>
struct newdoublegyre : field<Real, 2, 2> {
  using this_t   = newdoublegyre<Real>;
  using parent_t = field<Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using typename parent_t::symtensor_t;
  using parent_t::t;
  using parent_t::x;

  static GiNaC::numeric c() { return -0.2040811331; }
  static GiNaC::numeric d() { return 9.964223388; }

  //============================================================================
  newdoublegyre() {
    using GiNaC::Pi;
    GiNaC::numeric epsilon{1, 4};
    GiNaC::numeric A{1, 10};
    auto           omega = 2 * Pi * GiNaC::numeric{1, 10};
    auto           a     = epsilon * sin(omega * (t() + timeoffset()));
    auto           b     = 1 - 2 * a;
    auto           f     = a * x(0) * x(0) + b * x(0);
    auto           df    = 2 * a * x(0) + b;

    this->set_expr(symtensor_t{-Pi * A * sin(Pi * f) * cos(Pi * x(1)),
                                Pi * A * cos(Pi * f) * sin(Pi * x(1)) * df});
  }

  //----------------------------------------------------------------------------
  static auto timeoffset() {
    using GiNaC::Pi;
    auto r = Pi * GiNaC::numeric{1, 5} * t() + d();

    auto q = (4 * Pi * c() * sin(r) - 4 * asin(2 * c() * cos(r))) /
             (Pi * (1 - c() * c() * sin(r) * sin(r)));

    return 5 / Pi * asin(q) - t();
  }
  //----------------------------------------------------------------------------
  static auto timeoffset(Real time) {
    auto ex = timeoffset();
    return evtod<Real>(ex, t() == time);
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }

  //----------------------------------------------------------------------------
  static GiNaC::ex bifurcationline()  {
    using GiNaC::Pi;
    return c() * sin(Pi / 5 * t() + d()) + 1;
  }

  //----------------------------------------------------------------------------
  static auto bifurcationline(Real time)  {
    auto ex = bifurcationline();
    return evtod<Real>(ex, t() == time);
  }

  //----------------------------------------------------------------------------
  static auto bifurcationline(const linspace<Real>& domain) {
    auto bifu_ex = bifurcationline();
    parameterized_line<Real, 2> curve;
    for (auto time : domain) {
      curve.push_back({evtod<Real>(bifu_ex, t() == time), 0}, time);
    }
    return curve;
  }

  //----------------------------------------------------------------------------
  static auto bifurcationline_spacetime(const linspace<Real>& domain) {
    auto bifu_ex = bifurcationline();
    parameterized_line<Real, 3> curve;
    for (auto time : domain) {
      curve.push_back({evtod<Real>(bifu_ex, t() == time), 0, time}, time);
    }
    return curve;
  }
};

//==============================================================================
newdoublegyre() -> newdoublegyre<double>;

//==============================================================================
}  // namespace tatooine::analytical
//==============================================================================
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
//! Double Gyre dataset
template <typename Real>
struct newdoublegyre : field<newdoublegyre<Real>,Real, 2, 2> {
  using this_t   = newdoublegyre<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  static constexpr Real pi      = M_PI;
  static constexpr Real epsilon = 0.25;
  static constexpr Real omega   = 2 * pi * 0.1;
  static constexpr Real A       = 0.1;
  static constexpr Real c       = -0.2040811331;
  static constexpr Real cc      = c * c;
  static constexpr Real d       = 9.964223388;

  //============================================================================
  constexpr tensor_t evaluate(const pos_t& x, Real t) const {
    const Real a  = epsilon * std::sin(omega * (t + timeoffset(t)));
    const Real b  = 1 - 2 * a;
    const Real f  = a * x(0) * x(0) + b * x(0);
    const Real df = 2 * a * x(0) + b;

    return {-pi * A * std::sin(pi * f) * std::cos(pi * x(1)),
             pi * A * std::cos(pi * f) * std::sin(pi * x(1)) * df};
  }

  //----------------------------------------------------------------------------
  constexpr static auto timeoffset(const Real t) {
    const Real r = pi / 5 * t + d;

    const Real q =
        std::clamp<Real>((4 * pi * c * sin(r) - 4 * std::asin(2 * c * cos(r))) /
                             (pi * (1 - cc * sin(r) * sin(r))),
                         -1, 1);

    const Real p           = 5 / pi * std::asin(q) - t;
    return p;
    //Real       min_p       = p;
    //auto       closer_to_0 = [&min_p](Real p) -> Real {
    //  if (std::abs(p) < std::abs(min_p)) { return p; }
    //  return min_p;
    //};
    //
    //for (int i = 0; i <= 1; ++i) {
    //  min_p = closer_to_0(5 + i * 10 - 2 * t - p);
    //  min_p = closer_to_0(5 - i * 10 - 2 * t - p);
    //}
    //
    //for (int i = 1; i <= 1; ++i) {
    //  min_p = closer_to_0(p + i * 10);
    //  min_p = closer_to_0(p - i * 10);
    //}
    //
    //return min_p;
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }

  //----------------------------------------------------------------------------
  constexpr Real bifurcationline(Real t) const {
    return c * std::sin(pi / 5 * t + d) + 1;
  }

  //----------------------------------------------------------------------------
  constexpr auto bifurcationline(const linspace<Real>& domain) const {
    parameterized_line<Real, 2> c;
    for (auto t : domain) {
      c.push_back({bifurcationline_pos(t), 0}, t);
    }
    return c;
  }

  //----------------------------------------------------------------------------
  constexpr auto bifurcationline_spacetime(const linspace<Real>& domain) const {
    parameterized_line<Real, 3> c;
    for (auto t : domain) {
      c.push_back({bifurcationline_pos(t), 0, t}, t);
    }
    return c;
  }
};

//==============================================================================
newdoublegyre() -> newdoublegyre<double>;

//==============================================================================
}  // namespace tatooine::analytical
//==============================================================================

#endif
