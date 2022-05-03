#ifndef TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SINUSCOSINUS_H
#define TATOOINE_ANALYTICAL_FIELDS_NUMERICAL_SINUSCOSINUS_H
//==============================================================================
#include <tatooine/field.h>
#include <boost/math/constants/constants.hpp>
#include <cmath>
//==============================================================================
namespace tatooine::analytical::fields::numerical {
//==============================================================================
template <typename Real>
struct cosinussinus : vectorfield<cosinussinus<Real>, Real, 2> {
  using this_type   = cosinussinus<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  Real m_radius;
  //============================================================================
  constexpr cosinussinus(Real const r = 1) noexcept : m_radius{r} {}
  //----------------------------------------------------------------------------
  constexpr auto set_radius(Real r) { m_radius = r; }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_type& /*x*/, Real t) const
      -> tensor_type {
    return {gcem::cos(t) * m_radius, gcem::sin(t) * m_radius};
  }
};
//==============================================================================
template <typename Real>
struct sinuscosinus : vectorfield<sinuscosinus<Real>, Real, 2> {
  using this_type   = sinuscosinus<Real>;
  using parent_type = vectorfield<this_type, Real, 2>;
  using typename parent_type::pos_type;
  using typename parent_type::tensor_type;
  //============================================================================
  Real m_radius;
  //============================================================================
  constexpr sinuscosinus(Real const r = 1) noexcept : m_radius{r} {}
  //----------------------------------------------------------------------------
  constexpr auto set_radius(Real r) { m_radius = r; }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_type& /*x*/, Real t) const
      -> tensor_type {
    return {std::sin(t) * m_radius, std::cos(t) * m_radius};
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sinuscosinus()->sinuscosinus<double>;
cosinussinus()->cosinussinus<double>;
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#endif
