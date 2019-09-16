#ifndef TATOOINE_DOUBLEGYRE_H
#define TATOOINE_DOUBLEGYRE_H

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include "symbolic_field.h"

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================

template <typename Real>
struct doublegyre : field<Real, 2, 2> {
  using this_t   = doublegyre<Real>;
  using parent_t = field<Real, 2, 2>;
  using parent_t::t;
  using parent_t::x;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  doublegyre(const GiNaC::ex& eps   = GiNaC::numeric{1, 4},
             const GiNaC::ex& omega = 2 * GiNaC::Pi * GiNaC::numeric{1, 10},
             const GiNaC::ex& A     = GiNaC::numeric{1, 10}) {
    using GiNaC::Pi;
    auto a = eps * sin(omega * t());
    auto b = 1 - 2 * a;
    auto f = a * pow(x(0), 2) + b * x(0);
    this->set_expr(vec{
        -Pi * A * sin(Pi * f) * cos(Pi * x(1)),
        Pi * A * cos(Pi * f) * sin(Pi * x(1)) * f.diff(x(0))});
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }
};

doublegyre()->doublegyre<double>;

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================

//==============================================================================
namespace tatooine::numerical {
//==============================================================================
/// Double Gyre dataset
template <typename Real>
struct doublegyre : field<doublegyre<Real>, Real, 2, 2> {
  using this_t   = doublegyre<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  Real                epsilon, omega, A;
  static constexpr auto pi = boost::math::constants::pi<Real>();

  //============================================================================
  constexpr doublegyre(Real p_epsilon = 0.25, Real p_omega = 2 * pi * 0.1,
                       Real p_A = 0.1) noexcept
      : epsilon{p_epsilon}, omega{p_omega}, A{p_A} {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, Real t) const {
    Real a  = epsilon * sin(omega * t);
    Real b  = 1.0 - 2.0 * a;
    Real f  = a * x(0) * x(0) + b * x(0);
    Real df = 2 * a * x(0) + b;

    return {-pi * A * std::sin(pi * f) * std::cos(pi * x(1)),
             pi * A * std::cos(pi * f) * std::sin(pi * x(1)) * df};
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }
};

doublegyre()->doublegyre<double>;

//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
