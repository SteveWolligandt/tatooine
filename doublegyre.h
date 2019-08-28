#ifndef TATOOINE_DOUBLEGYRE_H
#define TATOOINE_DOUBLEGYRE_H

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include "symbolic_field.h"

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================

template <typename real_t>
struct doublegyre : field<real_t, 2, 2> {
  using this_t   = doublegyre<real_t>;
  using parent_t = field<real_t, 2, 2>;
  using parent_t::t;
  using parent_t::x;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  doublegyre(const GiNaC::ex& eps   = GiNaC::numeric{1, 4},
             const GiNaC::ex& omega = 2 * GiNaC::Pi / 10,
             const GiNaC::ex& A     = GiNaC::numeric{1, 10}) {
    using GiNaC::Pi;
    auto a = eps * sin(omega * t());
    auto b = 1 - 2 * a;
    auto f = a * pow(x(0), 2) + b * x(0);
    this->set_expr(
        symtensor_t{-Pi * A * sin(Pi * f) * cos(Pi * x(1)),
                    Pi * A * cos(Pi * f) * sin(Pi * x(1)) * f.diff(x(0))});
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t) const {
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
template <typename real_t>
struct doublegyre : field<doublegyre<real_t>, real_t, 2, 2> {
  using this_t   = doublegyre<real_t>;
  using parent_t = field<this_t, real_t, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  real_t                epsilon, omega, A;
  static constexpr auto pi = boost::math::constants::pi<real_t>();

  //============================================================================
  constexpr doublegyre(real_t p_epsilon = 0.25, real_t p_omega = 2 * pi * 0.1,
                       real_t p_A = 0.1) noexcept
      : epsilon{p_epsilon}, omega{p_omega}, A{p_A} {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, real_t t) const {
    real_t a  = epsilon * sin(omega * t);
    real_t b  = 1.0 - 2.0 * a;
    real_t f  = a * x(0) * x(0) + b * x(0);
    real_t df = 2 * a * x(0) + b;

    return {-pi * A * std::sin(pi * f) * std::cos(pi * x(1)),
            pi * A * std::cos(pi * f) * std::sin(pi * x(1)) * df};
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, real_t) const {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }
};

doublegyre()->doublegyre<double>;

//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
