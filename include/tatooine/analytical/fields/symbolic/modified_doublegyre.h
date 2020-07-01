#ifndef TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_MODIFIED_DOUBLEGYRE_H
#define TATOOINE_ANALYTICAL_FIELDS_SYMBOLIC_MODIFIED_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/packages.h>
#if TATOOINE_GINAC_AVAILABLE
#include <tatooine/symbolic_field.h>
//==============================================================================
namespace tatooine::analytical::fields::symbolic {
//==============================================================================
template <typename Real>
struct modified_doublegyre : field<Real, 2, 2> {
  using this_t   = modified_doublegyre<Real>;
  using parent_t = field<Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  using typename parent_t::symtensor_t;
  using parent_t::t;
  using parent_t::x;

  static GiNaC::numeric c() { return -0.2040811331; }
  static GiNaC::numeric d() { return 9.964223388; }

  //============================================================================
  modified_doublegyre() {
    using GiNaC::Pi;
    GiNaC::numeric epsilon{1, 4};
    GiNaC::numeric A{1, 10};
    auto           omega = 2 * Pi * GiNaC::numeric{1, 10};
    auto           a     = epsilon * sin(omega * (t() + timeoffset()));
    auto           b     = 1 - 2 * a;
    auto           f     = a * x(0) * x(0) + b * x(0);
    auto           df    = 2 * a * x(0) + b;

    this->set_expr(vec{-Pi * A * sin(Pi * f) * cos(Pi * x(1)),
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
modified_doublegyre() -> modified_doublegyre<double>;
//==============================================================================
}  // namespace tatooine::analytical
//==============================================================================
#endif
#endif
