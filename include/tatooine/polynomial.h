#ifndef TATOOINE_POLYNOMIAL_H
#define TATOOINE_POLYNOMIAL_H
//==============================================================================
#include <array>
#include <ostream>

#include "make_array.h"
#include "tensor.h"
#include "type_traits.h"
//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real, size_t NumDegrees>
struct polynomial {
  //----------------------------------------------------------------------------
  // static methods
  //----------------------------------------------------------------------------
 public:
  static constexpr size_t num_degrees() { return NumDegrees; }
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  std::array<Real, NumDegrees + 1> m_coefficients;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr polynomial(const std::array<Real, NumDegrees + 1>& coeffs)
      : m_coefficients{coeffs} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(std::array<Real, NumDegrees + 1>&& coeffs)
      : m_coefficients{std::move(coeffs)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Coeffs, enable_if_arithmetic<Coeffs...> = true,
            std::enable_if_t<sizeof...(Coeffs) == NumDegrees + 1, bool> = true>
  constexpr polynomial(Coeffs... coeffs)
      : m_coefficients{static_cast<Real>(coeffs)...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr polynomial(const vec<OtherReal, NumDegrees + 1>& coeffs)
      : m_coefficients{make_array<Real>(coeffs.data())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr polynomial(const std::array<OtherReal, NumDegrees + 1>& coeffs)
      : m_coefficients{make_array<Real>(coeffs)} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  /// evaluates c0 * x^0 + c1 * x^1 + ... + c{N-1} * x^{N-1}
  constexpr auto evaluate(Real x) const {
    Real y   = 0;
    Real acc = 1;

    for (size_t i = 0; i < NumDegrees + 1; ++i) {
      y += acc * m_coefficients[i];
      acc *= x;
    }

    return y;
  }
  //----------------------------------------------------------------------------
  /// evaluates c0 * x^0 + c1 * x^1 + ... + c{N-1} * x^{N-1}
  constexpr auto operator()(Real x) const { return evaluate(x); }
  //----------------------------------------------------------------------------
  auto&       c(size_t i) { return m_coefficients[i]; }
  const auto& c(size_t i) const { return m_coefficients[i]; }
  //----------------------------------------------------------------------------
  auto&       coefficient(size_t i) { return m_coefficients[i]; }
  const auto& coefficient(size_t i) const { return m_coefficients[i]; }
  //----------------------------------------------------------------------------
  auto&       coefficients() { return m_coefficients; }
  const auto& coefficients() const { return m_coefficients; }
  //----------------------------------------------------------------------------
  constexpr void set_coefficients(
      const std::array<Real, NumDegrees + 1>& coeffs) {
    m_coefficients = coeffs;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr void set_coefficients(std::array<Real, NumDegrees + 1>&& coeffs) {
    m_coefficients = std::move(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr void set_coefficients(
      const std::array<OtherReal, NumDegrees + 1>& coeffs) {
    m_coefficients = make_array<Real>(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Coeffs, enable_if_arithmetic<Coeffs...> = true,
            std::enable_if_t<sizeof...(Coeffs) == NumDegrees + 1, bool> = true>
  constexpr void set_coefficients(Coeffs... coeffs) {
    m_coefficients =
        std::array<Real, NumDegrees + 1>{static_cast<Real>(coeffs)...};
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto diff(std::index_sequence<Is...>) const {
    return polynomial<Real, NumDegrees - 1>{
        (m_coefficients[Is + 1] * (Is + 1))...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  template <size_t... Is>
  constexpr auto diff() const {
    if constexpr (NumDegrees >= 1) {
      return diff(std::make_index_sequence<NumDegrees>{});
    } else {
      return polynomial<Real, 0>{0};
    }
  }
};

//------------------------------------------------------------------------------
// deduction guides
//------------------------------------------------------------------------------
template <typename... Coeffs>
polynomial(Coeffs... coeffs)
    ->polynomial<promote_t<Coeffs...>, sizeof...(Coeffs) - 1>;

//------------------------------------------------------------------------------
// diff
//------------------------------------------------------------------------------
template <typename Real, size_t NumDegrees>
constexpr auto diff(const polynomial<Real, NumDegrees>& f) {
  return f.diff();
}

//------------------------------------------------------------------------------
// I/O
//------------------------------------------------------------------------------
template <typename Real, size_t NumDegrees>
auto& operator<<(std::ostream& out, const polynomial<Real, NumDegrees>& f) {
  out << f.c(0);
  for (size_t i = 1; i < NumDegrees + 1; ++i) {
    out << " + " << f.c(i) << " * x^" << i;
  }
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
