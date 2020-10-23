#ifndef TATOOINE_DUFFING_OSCILLATOR_H
#define TATOOINE_DUFFING_OSCILLATOR_H
//==============================================================================
#include <tatooine/field.h>

#include <boost/math/constants/constants.hpp>
#include <cmath>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <typename Real>
struct duffing_oscillator : vectorfield<duffing_oscillator<Real>, Real, 2> {
  using this_t   = duffing_oscillator<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  Real                  m_delta, m_alpha, m_beta;
  static constexpr auto pi = boost::math::constants::pi<Real>();
  //============================================================================
  constexpr duffing_oscillator(Real delta, Real alpha, Real beta) noexcept
      : m_delta{delta}, m_alpha{alpha}, m_beta{beta} {}
  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, Real t) const override {
    return tensor_t{
        x(1), -m_delta * x(1) - m_alpha * x(0) - m_beta * x(0) * x(0) * x(0)};
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const override {
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
  using this_t   = forced_duffing_oscillator<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  Real                  m_eps;
  static constexpr auto pi = boost::math::constants::pi<Real>();
  //============================================================================
  constexpr forced_duffing_oscillator(Real eps = 0.25) noexcept : m_eps{eps} {}
  ~forced_duffing_oscillator() override = default;
  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, Real t) const override {
    return {x(1), x(0) - x(0) * x(0) * x(0) + m_eps * std::sin(t)};
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const override {
    return true;
  }
};
//------------------------------------------------------------------------------
forced_duffing_oscillator()->forced_duffing_oscillator<double>;
//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
