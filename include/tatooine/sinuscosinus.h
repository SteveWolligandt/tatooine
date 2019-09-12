#ifndef TATOOINE_SINUSCOSINUS_H
#define TATOOINE_SINUSCOSINUS_H

#include <boost/math/constants/constants.hpp>
#include <cmath>
#include "symbolic_field.h"

//==============================================================================
namespace tatooine::symbolic {
//==============================================================================

template <typename real_t>
struct cosinussinus : field<real_t, 2, 2> {
  using this_t   = cosinussinus<real_t>;
  using parent_t = field<real_t, 2, 2>;
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
template <typename real_t>
struct sinuscosinus : field<real_t, 2, 2> {
  using this_t   = sinuscosinus<real_t>;
  using parent_t = field<real_t, 2, 2>;
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

template <typename real_t>
struct cosinussinus : field<cosinussinus<real_t>, real_t, 2, 2> {
  using this_t   = cosinussinus<real_t>;
  using parent_t = field<this_t, real_t, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  constexpr cosinussinus() noexcept {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& /*x*/, real_t t) const {
    return {std::sin(t), std::cos(t)};
  }
};

//==============================================================================
template <typename real_t>
struct sinuscosinus : field<sinuscosinus<real_t>, real_t, 2, 2> {
  using this_t   = sinuscosinus<real_t>;
  using parent_t = field<this_t, real_t, 2, 2>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;

  //============================================================================
  constexpr sinuscosinus() noexcept {}

  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& /*x*/, real_t t) const {
    return {std::sin(t), std::cos(t)};
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sinuscosinus()->sinuscosinus<double>;
cosinussinus()->cosinussinus<double>;

//==============================================================================
}  // namespace tatooine::numerical
//==============================================================================
#endif
