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
struct cosinussinus : field<cosinussinus<Real>, Real, 2, 2> {
  using this_t   = cosinussinus<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
  Real m_radius;
  //============================================================================
  constexpr cosinussinus(Real r = 0.5) noexcept : m_radius{r} {}
  //----------------------------------------------------------------------------
  void set_radius(Real r) { m_radius = r; }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t& /*x*/, Real t) const
      -> tensor_t final {
    return tensor_t{std::cos(t) * m_radius, std::sin(t) * m_radius};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(const pos_t& /*x*/, Real /*t*/) const
      -> bool final {
    return true;
  }
};
//==============================================================================
template <typename Real>
struct sinuscosinus : field<sinuscosinus<Real>, Real, 2, 2> {
  using this_t   = sinuscosinus<Real>;
  using parent_t = field<this_t, Real, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  constexpr sinuscosinus() noexcept {}
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto evaluate(const pos_t& /*x*/, Real t) const
      -> tensor_t final {
    return tensor_t{std::sin(t), std::cos(t)};
  }
  //----------------------------------------------------------------------------
  [[nodiscard]] constexpr auto in_domain(const pos_t& /*x*/, Real /*t*/) const
      -> bool final {
    return true;
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sinuscosinus()->sinuscosinus<double>;
cosinussinus()->cosinussinus<double>;
//==============================================================================
}  // namespace tatooine::analytical::fields::numerical
//==============================================================================
#endif
