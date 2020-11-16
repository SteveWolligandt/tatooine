#ifndef TATOOINE_POLYNOMIAL_H
#define TATOOINE_POLYNOMIAL_H
//==============================================================================
#include <array>
#include <ostream>
#include <type_traits>

#include "make_array.h"
#include "linspace.h"
#include "tensor.h"
#include "type_traits.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Real, size_t Degree>
struct polynomial {
  //----------------------------------------------------------------------------
  // typedefs
  //----------------------------------------------------------------------------
 public:
   using real_t = Real;
  //----------------------------------------------------------------------------
  // static methods
  //----------------------------------------------------------------------------
 public:
  static constexpr size_t degree() { return Degree; }
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 private:
  std::array<Real, Degree + 1> m_coefficients;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
 public:
  constexpr polynomial() : m_coefficients{make_array<Real, Degree + 1>()} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(const polynomial& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(polynomial&& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial& operator=(const polynomial& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial& operator=(polynomial&& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(const std::array<Real, Degree + 1>& coeffs)
      : m_coefficients{coeffs} {}
  //----------------------------------------------------------------------------
  template <typename OtherReal, size_t OtherDegree,
            std::enable_if_t<(OtherDegree <= Degree), bool> = true>
  constexpr polynomial(const polynomial<OtherReal, OtherDegree>& other)
      : m_coefficients{make_array<Real, Degree + 1>(0)} {
    for (size_t i = 0; i < OtherDegree + 1; ++i) {
      m_coefficients[i] = other.coefficient(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(std::array<Real, Degree + 1>&& coeffs)
      : m_coefficients{std::move(coeffs)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Coeffs, enable_if_arithmetic<Coeffs...> = true,
            std::enable_if_t<sizeof...(Coeffs) == Degree + 1, bool> = true>
  constexpr polynomial(Coeffs... coeffs)
      : m_coefficients{static_cast<Real>(coeffs)...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr polynomial(const vec<OtherReal, Degree + 1>& coeffs)
      : m_coefficients{make_array<Real>(coeffs.data())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr polynomial(const std::array<OtherReal, Degree + 1>& coeffs)
      : m_coefficients{make_array<Real>(coeffs)} {}

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
 public:
  /// evaluates c0 * x^0 + c1 * x^1 + ... + c{N-1} * x^{N-1}
  constexpr auto evaluate(Real x) const {
    Real y   = 0;
    Real acc = 1;
    for (size_t i = 0; i < Degree + 1; ++i) {
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
      const std::array<Real, Degree + 1>& coeffs) {
    m_coefficients = coeffs;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr void set_coefficients(std::array<Real, Degree + 1>&& coeffs) {
    m_coefficients = std::move(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
  constexpr void set_coefficients(
      const std::array<OtherReal, Degree + 1>& coeffs) {
    m_coefficients = make_array<Real>(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Coeffs, enable_if_arithmetic<Coeffs...> = true,
            std::enable_if_t<sizeof...(Coeffs) == Degree + 1, bool> = true>
  constexpr void set_coefficients(Coeffs... coeffs) {
    m_coefficients =
        std::array<Real, Degree + 1>{static_cast<Real>(coeffs)...};
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto diff(std::index_sequence<Is...> /*seq*/) const {
    return polynomial<Real, Degree - 1>{
        (m_coefficients[Is + 1] * (Is + 1))...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 public:
  constexpr auto diff() const {
    if constexpr (Degree >= 1) {
      return diff(std::make_index_sequence<Degree>{});
    } else {
      return polynomial<Real, 0>{0};
    }
  }
  auto print(std::ostream& out, const std::string& x) const -> std::ostream& {
    out << c(0);
    if (Degree >= 1) {
      if (c(1) != 0) {
        if (c(1) == 1) {
          out << " + " << x;
        }else if (c(1) == -1) {
          out << " - " << x;
        } else {
          out << " + " << c(1) << " * " << x;
        }
      }
    }
    for (size_t i = 2; i < Degree + 1; ++i) {
      if (c(i) != 0) {
        if (c(i) == 1) {
          out << " + " << x << "^" << i;
        } else if (c(i) == -1) {
          out << " - " << x << "^" << i;
        } else {
          out << " + " << c(i) << " * " << x << "^" << i;
        }
      }
    }
    return out;
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
template <typename Real, size_t Degree>
constexpr auto diff(const polynomial<Real, Degree>& f) {
  return f.diff();
}
//------------------------------------------------------------------------------
// solve
//------------------------------------------------------------------------------
/// solve a + bx
template <typename Real>
auto solve(const polynomial<Real, 1>& p) -> std::vector<Real> {
  if (p.c(1) == 0) { return {}; }
  return {-p.c(0) / p.c(1)};
}
//------------------------------------------------------------------------------
/// solve a + bx + cxx
template <typename Real>
auto solve(const polynomial<Real, 2>& p) -> std::vector<Real> {
  auto const a = p.c(0);
  auto const b = p.c(1);
  auto const c = p.c(2);
  if (c == 0) {
    return solve(polynomial{a, b});
  }

  auto const discr = b * b - 4 * a * c;
  if (discr < 0) {
    return {};
  } else if (std::abs(discr) < 1e-10) {
    return {-b / (2 * c)};
  }
  std::vector<Real> solutions;
  solutions.reserve(2);
  Real q = (b > 0) ? -0.5 * (b + sqrt(discr)) : -0.5 * (b - sqrt(discr));
  solutions.push_back(q / c);
  solutions.push_back(a / q);
  std::swap(solutions[0], solutions[1]);
  return solutions;
}
//------------------------------------------------------------------------------
// I/O
//------------------------------------------------------------------------------
template <typename Real, size_t Degree>
auto& operator<<(std::ostream& out, const polynomial<Real, Degree>& f) {
  return f.print(out, "x");
}
//------------------------------------------------------------------------------
// type_traits
//------------------------------------------------------------------------------
template <typename T>
struct is_polynomial : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t Degree>
struct is_polynomial<polynomial<Real, Degree>> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr bool is_polynomial_v = is_polynomial<T>::value;
//------------------------------------------------------------------------------
template <typename... Ts>
struct are_polynomial;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto are_polynomial_v = are_polynomial<Ts...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <>
struct are_polynomial<> : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct are_polynomial<T>
    : std::integral_constant<bool, is_polynomial_v<T>> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T0, typename T1, typename... Ts>
struct are_polynomial<T0, T1, Ts...>
    : std::integral_constant<bool, are_polynomial_v<T0> &&
                                       are_polynomial_v<T1, Ts...>> {};
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_polynomial =
    std::enable_if_t<sizeof...(Ts) == 0 || are_polynomial_v<Ts...>, bool>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
