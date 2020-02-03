#ifndef TATOOINE_POLYNOMIAL_H
#define TATOOINE_POLYNOMIAL_H
//==============================================================================
#include <array>
#include "make_array.h"
#include "type_traits.h"
//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t Degrees>
struct polynomial {
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  std::array<Real, Degrees> m_coefficients;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr polynomial(const std::array<Real, N>& coeffs)
      : m_coefficients{coeffs} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(std::array<Real, N>&& coeffs)
      : m_coefficients{std::move(coeffs)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Coeffs, enable_if_arithmetic<Coeffs...> = true>
  constexpr polynomial(Coeffs... coeffs) : m_coefficients{coeffs...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr polynomial(const vec<OtherReal, N>& coeffs)
      : m_coefficients{make_array<Real>(coeffs)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr polynomial(const std::array<OtherReal, N>& coeffs)
      : m_coefficients{make_array<Real>(coeffs)} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  /// evaluates c0 * x^0 + c1 * x^1 + ... + c{N-1} * x^{N-1}
  constexpr auto evaluate(Real x) const {
    Real y   = 0;
    Real acc = 1;

    for (size_t i = 0; i < N; ++i) {
      y += acc * m_coefficients[i];
      acc *= x;
    }

    return y;
  }
  //----------------------------------------------------------------------------
  /// evaluates c0 * x^0 + c1 * x^1 + ... + c{N-1} * x^{N-1}
  constexpr auto operator()(Real x) const { return evaluate(x); }
};

//------------------------------------------------------------------------------
// deduction guides
//------------------------------------------------------------------------------
template <typename... Coeffs>
polynomial(Coeffs... coeffs)
    ->polynomial<promote_t<Coeffs...>, sizeof...(Coeffs)>;
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
