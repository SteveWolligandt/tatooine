#ifndef TATOOINE_DUFFING_OSCILLATOR_H
#define TATOOINE_DUFFING_OSCILLATOR_H

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include "field.h"

//==============================================================================
namespace tatooine {
namespace numerical {
//==============================================================================
template <typename Real>
struct duffing_oscillator : field<duffing_oscillator<Real>, Real, 2, 2> {
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
  constexpr tensor_t evaluate(const pos_t& x, Real t) const {
    return {x(1), -m_delta * x(1) - m_alpha*x(0) - m_beta * x(0) * x(0) * x(0)};
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const {
    return true;
  }
};

#if has_cxx17_support()
duffing_oscillator()->duffing_oscillator<double>;
#endif

//==============================================================================
template <typename Real>
struct forced_duffing_oscillator : field<forced_duffing_oscillator<Real>, Real, 2, 2> {
  using this_t   = forced_duffing_oscillator<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  Real                  m_eps;
  static constexpr auto pi = boost::math::constants::pi<Real>();

  //============================================================================
  constexpr forced_duffing_oscillator(Real eps = 0.25) noexcept : m_eps{eps} {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, Real t) const {
    return {x(1), x(0) - x(0) * x(0) * x(0) + m_eps * std::sin(t)};
  }

  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& x, Real) const {
    return true;
  }
};

#if has_cxx17_support()
forced_duffing_oscillator()->forced_duffing_oscillator<double>;
#endif

//==============================================================================
}  // namespace numerical
}  // namespace tatooine
//==============================================================================
#endif
