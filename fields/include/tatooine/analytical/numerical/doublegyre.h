#ifndef TATOOINE_ANALYTICAL_NUMERICAL_DOUBLEGYRE_H
#define TATOOINE_ANALYTICAL_NUMERICAL_DOUBLEGYRE_H
//==============================================================================
#include <tatooine/field.h>
#include <cmath>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
/// Double Gyre dataset
template <typename Real>
struct doublegyre : vectorfield<doublegyre<Real>, Real, 2> {
  using this_type   = doublegyre<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  //============================================================================
  static constexpr auto pi = std::numbers::template pi_v<Real>;
  //============================================================================
  Real m_epsilon, m_omega, m_A;
  bool m_infinite_domain = false;
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
  [[nodiscard]] constexpr auto evaluate(
      fixed_size_vec<2> auto const& x, Real const t) const
      -> tensor_type {
    if (!m_infinite_domain && (x(0) < 0 || x(0) > 2 || x(1) < 0 || x(1) > 1)) {
      return parent_type::ood_tensor();
    }
    Real const a  = m_epsilon * gcem::sin(m_omega * t);
    Real const b  = 1 - 2 * a;
    Real const f  = a * x(0) * x(0) + b * x(0);
    Real const df = 2 * a * x(0) + b;

    return {-pi * m_A * gcem::sin(pi * f) * gcem::cos(pi * x(1)),
            pi * m_A * gcem::cos(pi * f) * gcem::sin(pi * x(1)) * df};
  }
  //----------------------------------------------------------------------------
  constexpr auto set_infinite_domain(bool const v = true) {
    m_infinite_domain = v;
  }
  //----------------------------------------------------------------------------
  constexpr auto epsilon() const { return m_epsilon; }
  constexpr auto epsilon() -> auto& { return m_epsilon; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_epsilon(Real const epsilon) { m_epsilon = epsilon; }
  //----------------------------------------------------------------------------
  constexpr auto omega() const { return m_omega; }
  constexpr auto omega() -> auto& { return m_omega; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_omega(Real const omega) { m_omega = omega; }
  //----------------------------------------------------------------------------
  constexpr auto A() const { return m_A; }
  constexpr auto A() -> auto& { return m_A; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_A(Real const A) { m_A = A; }
};
//==============================================================================
doublegyre()->doublegyre<real_number>;
//==============================================================================
}  // namespace tatooine::analytical::numerical
//==============================================================================
#endif
