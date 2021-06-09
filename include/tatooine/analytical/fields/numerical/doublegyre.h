#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_DOUBLEGYRE_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/field.h>

#define _USE_MATH_DEFINES
#include <cmath>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
/// Double Gyre dataset
template <typename Real>
struct doublegyre : vectorfield<doublegyre<Real>, Real, 2> {
  using this_t   = doublegyre<Real>;
  using parent_t = vectorfield<this_t, Real, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::real_t;
  using typename parent_t::tensor_t;
  //============================================================================
  static constexpr auto pi = M_PI;
  //============================================================================
  Real m_epsilon, m_omega, m_A;
  bool m_infinite_domain  = false;
  //============================================================================
  explicit constexpr doublegyre(Real const epsilon = 0.25,
                                Real const omega   = 2 * pi * 0.1,
                                Real const A       = 0.1) noexcept
      : m_epsilon{epsilon}, m_omega{omega}, m_A{A} {}
  //------------------------------------------------------------------------------
  constexpr doublegyre(doublegyre const&)     = default;
  constexpr doublegyre(doublegyre&&) noexcept = default;
  //------------------------------------------------------------------------------
  constexpr auto operator=(doublegyre const&) -> doublegyre& = default;
  constexpr auto operator=(doublegyre&&) noexcept -> doublegyre& = default;
  //------------------------------------------------------------------------------
  ~doublegyre() override = default;
  //----------------------------------------------------------------------------
  [[nodiscard]] auto evaluate(pos_t const& x, Real const t) const
      -> tensor_t {
    Real const a  = m_epsilon * sin(m_omega * t);
    Real const b  = 1.0 - 2.0 * a;
    Real const f  = a * x(0) * x(0) + b * x(0);
    Real const df = 2 * a * x(0) + b;

    return {-pi * m_A * std::sin(pi * f) * std::cos(pi * x(1)),
            pi * m_A * std::cos(pi * f) * std::sin(pi * x(1)) * df};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] auto in_domain(pos_t const& x, Real const /*t*/) const
      -> bool {
    if (m_infinite_domain) {
      return true;
    }
    return 0 <= x(0) && x(0) <= 2 && 0 <= x(1) && x(1) <= 1;
  }
  //----------------------------------------------------------------------------
  constexpr void set_infinite_domain(bool const v = true) {
    m_infinite_domain = v;
  }
  //----------------------------------------------------------------------------
  constexpr auto epsilon() const { return m_epsilon; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto epsilon() -> auto& { return m_epsilon; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_epsilon(Real const epsilon) { m_epsilon = epsilon; }
  //----------------------------------------------------------------------------
  constexpr auto omega() const { return m_omega; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto omega() -> auto& { return m_omega; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_omega(Real const omega) { m_omega = omega; }
  //----------------------------------------------------------------------------
  constexpr auto A() const { return m_A; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto A() -> auto& { return m_A; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_A(Real const A) { m_A = A; }
};

doublegyre()->doublegyre<double>;
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#endif
