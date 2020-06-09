#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_DOUBLEGYRE_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_DOUBLEGYRE_H
//==============================================================================
#include <boost/math/constants/constants.hpp>
#include <cmath>

#include <tatooine/field.h>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
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
  [[nodiscard]] constexpr auto in_domain(const pos_t& x, Real /*t*/) const
      -> bool final {
    if (m_infinite_domain) { return true; }
    return 0 <= x(0) && x(0) <= 2 && 0 <= x(1) && x(1) <= 1;
  }
  //----------------------------------------------------------------------------
  constexpr void set_infinite_domain(bool v) { m_infinite_domain = v; }
};

doublegyre()->doublegyre<double>;
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#endif
