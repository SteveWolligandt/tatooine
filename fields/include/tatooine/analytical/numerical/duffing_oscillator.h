#ifndef TATOOINE_DUFFING_OSCILLATOR_H
#define TATOOINE_DUFFING_OSCILLATOR_H
//==============================================================================
#include <tatooine/field.h>

#include <boost/math/constants/constants.hpp>
#include <cmath>
//==============================================================================
namespace tatooine::analytical::numerical {
//==============================================================================
template <typename Real>
struct duffing_oscillator : vectorfield<duffing_oscillator<Real>, Real, 2> {
  using this_type   = duffing_oscillator<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  Real                  m_delta, m_alpha, m_beta;
  static constexpr auto pi = boost::math::constants::pi<Real>();
  //============================================================================
  constexpr duffing_oscillator(Real delta, Real alpha, Real beta) noexcept
      : m_delta{delta}, m_alpha{alpha}, m_beta{beta} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(const pos_type& x, Real /*t*/) const
      -> tensor_type override {
    return tensor_type{
        x(1), -m_delta * x(1) - m_alpha * x(0) - m_beta * x(0) * x(0) * x(0)};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_type& /*x*/, Real /*t*/) const
      -> bool override {
    return true;
  }
  //----------------------------------------------------------------------------
  auto alpha() -> auto& { return m_alpha; }
  auto alpha() const { return m_alpha; }
  //----------------------------------------------------------------------------
  auto beta() -> auto& { return m_beta; }
  auto beta() const { return m_beta; }
  //----------------------------------------------------------------------------
  auto delta() -> auto& { return m_delta; }
  auto delta() const { return m_delta; }
};
//------------------------------------------------------------------------------
duffing_oscillator()->duffing_oscillator<double>;
//==============================================================================
template <typename Real>
struct forced_duffing_oscillator
    : vectorfield<forced_duffing_oscillator<Real>, Real, 2> {
  using this_type   = forced_duffing_oscillator<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  Real                  m_eps;
  static constexpr auto pi = boost::math::constants::pi<Real>();
  //============================================================================
  constexpr forced_duffing_oscillator(Real eps = 0.25) noexcept : m_eps{eps} {}
  ~forced_duffing_oscillator() override = default;
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& x, Real const t) const -> tensor_type {
    return {x(1), x(0) - x(0) * x(0) * x(0) + m_eps * std::sin(t)};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_type const& /*x*/, Real const /*t*/) const -> bool {
    return true;
  }
};
//------------------------------------------------------------------------------
forced_duffing_oscillator()->forced_duffing_oscillator<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
