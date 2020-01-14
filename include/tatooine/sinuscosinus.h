#ifndef TATOOINE_SINUSCOSINUS_H
#define TATOOINE_SINUSCOSINUS_H

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include "symbolic_field.h"

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================

template <typename Real>
struct cosinussinus : field<Real, 2, 2> {
  using this_t   = cosinussinus<Real>;
  using parent_t = field<Real, 2, 2>;
  using parent_t::t;
  using parent_t::x;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  cosinussinus() {
    this->set_expr(vec{GiNaC::ex{cos(t())}, GiNaC::ex{sin(t())}});
  }
};

//==============================================================================
template <typename Real>
struct sinuscosinus : field<Real, 2, 2> {
  using this_t   = sinuscosinus<Real>;
  using parent_t = field<Real, 2, 2>;
  using parent_t::t;
  using parent_t::x;
  using typename parent_t::pos_t;
  using typename parent_t::symtensor_t;
  using typename parent_t::tensor_t;

  sinuscosinus() {
    this->set_expr(vec{GiNaC::ex{sin(t())}, GiNaC::ex{cos(t())}});
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sinuscosinus()->sinuscosinus<double>;
cosinussinus()->cosinussinus<double>;

//==============================================================================
}  // namespace tatooine::symbolic
//==============================================================================

//==============================================================================
namespace tatooine::numerical {
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
  constexpr tensor_t evaluate(const pos_t& /*x*/, Real t) const {
    return {std::cos(t) * m_radius, std::sin(t) * m_radius};
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& /*x*/, Real /*t*/) const {
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
  constexpr tensor_t evaluate(const pos_t& /*x*/, Real t) const {
    return {std::sin(t), std::cos(t)};
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& /*x*/, Real /*t*/) const {
    return true;
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sinuscosinus()->sinuscosinus<double>;
cosinussinus()->cosinussinus<double>;

//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
