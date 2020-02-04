#ifndef TATOOINE_INTERPOLATION_H
#define TATOOINE_INTERPOLATION_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include "tensor.h"
#include "polynomial.h"

//==============================================================================
namespace tatooine::interpolation {
//==============================================================================
template <typename Data>
struct hermite {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = true;

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  Data m_fx0, m_fx1;
  Data m_fx0dx, m_fx1dx;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr hermite(const Data& fx0, const Data& fx1, const Data& fx0dx,
                    const Data& fx1dx)
      : m_fx0{fx0}, m_fx1{fx1}, m_fx0dx{fx0dx}, m_fx1dx{fx1dx} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  template <typename Real, enable_if_arithmetic<Real> = true>
  constexpr auto evaluate(Real t) const {
    return (2 * t * t * t - 3 * t * t + 1) * m_fx0 +
           (-2 * t * t * t + 3 * t * t) * m_fx1 +
           (t * t * t - 2 * t * t + t) * m_fx0dx +
           (t * t * t - t * t) * m_fx1dx;
  }
  template <typename Real, enable_if_arithmetic<Real> = true>
  constexpr auto operator()(Real t) const {
    return evaluate(t);
  }
};
//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================

#endif
