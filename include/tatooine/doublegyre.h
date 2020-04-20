#ifndef TATOOINE_DOUBLEGYRE_H
#define TATOOINE_DOUBLEGYRE_H

#include <boost/math/constants/constants.hpp>
#include <cmath>

#include "field.h"
#include "symbolic_field.h"
//==============================================================================
namespace tatooine::numerical {
//==============================================================================
/// Double Gyre dataset
template <typename Real>
struct doublegyre : vectorfield<doublegyre<Real>, Real, 2> {
  using this_t   = doublegyre<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  static constexpr auto pi = boost::math::constants::pi<Real>();
  //============================================================================
  Real m_epsilon, m_omega, m_A;
  bool m_infinite_domain;
  //============================================================================
  explicit constexpr doublegyre(Real epsilon = 0.25, Real omega = 2 * pi * 0.1,
                       Real A = 0.1) noexcept
      : m_epsilon{epsilon}, m_omega{omega}, m_A{A}, m_infinite_domain{false} {}
  constexpr doublegyre(const doublegyre&)     = default;
  constexpr doublegyre(doublegyre&&) noexcept = default;
  constexpr auto operator=(const doublegyre&) -> doublegyre& = default;
  constexpr auto operator=(doublegyre&&) noexcept -> doublegyre& = default;
  //----------------------------------------------------------------------------
  ~doublegyre() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t& x, Real t) const
      -> tensor_t final {
    Real a  = m_epsilon * sin(m_omega * t);
    Real b  = 1.0 - 2.0 * a;
    Real f  = a * x(0) * x(0) + b * x(0);
    Real df = 2 * a * x(0) + b;

    return tensor_t{-pi * m_A * std::sin(pi * f) * std::cos(pi * x(1)),
                     pi * m_A * std::cos(pi * f) * std::sin(pi * x(1)) * df};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(const pos_t& x, Real t) const
      -> bool final {
    return m_infinite_domain || (0 <= x(0) && x(0) <= 2 && 
                                 0 <= x(1) && x(1) <= 1 &&
                                 0 <= t    && t    <= 10);
  }
  //----------------------------------------------------------------------------
  constexpr void set_infinite_domain(bool v) { m_infinite_domain = v; }
};

doublegyre()->doublegyre<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================

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

  explicit doublegyre(const GiNaC::ex& eps   = GiNaC::numeric{1, 4},
                      const GiNaC::ex& omega = 2 * GiNaC::Pi *
                                               GiNaC::numeric{1, 10},
                      const GiNaC::ex& A = GiNaC::numeric{1, 10}) {
    using GiNaC::Pi;
    auto a = eps * sin(omega * t());
    auto b = 1 - 2 * a;
    auto f = a * pow(x(0), 2) + b * x(0);
    this->set_expr(vec<GiNaC::ex, 2>{
        -Pi * A * sin(Pi * f) * cos(Pi * x(1)),
        Pi * A * cos(Pi * f) * sin(Pi * x(1)) * f.diff(x(0))});
  }

  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& x, Real) const -> bool {
    return x(0) >= 0 && x(0) <= 2 && x(1) >= 0 && x(1) <= 1;
  }
};
doublegyre()->doublegyre<double>;

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================
#endif
