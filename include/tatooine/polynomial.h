#ifndef TATOOINE_POLYNOMIAL_H
#define TATOOINE_POLYNOMIAL_H
//==============================================================================
#include <array>
#include <ostream>
#include <type_traits>

#include <tatooine/make_array.h>
#include <tatooine/linspace.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>
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
  constexpr polynomial(polynomial const& other) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(polynomial&& other) noexcept = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator=(polynomial const& other)
      -> polynomial& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator=(polynomial&& other) noexcept
      -> polynomial& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(std::array<Real, Degree + 1> const& coeffs)
      : m_coefficients{coeffs} {}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename OtherReal, size_t OtherDegree>
  requires (OtherDegree <= Degree)
#else
  template <typename OtherReal, size_t OtherDegree,
            enable_if<(OtherDegree <= Degree)> = true>
#endif
  constexpr polynomial(polynomial<OtherReal, OtherDegree> const& other)
      : m_coefficients{make_array<Degree + 1>(Real(0))} {
    for (size_t i = 0; i < OtherDegree + 1; ++i) {
      m_coefficients[i] = other.coefficient(i);
    }
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(std::array<Real, Degree + 1>&& coeffs)
      : m_coefficients{std::move(coeffs)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic... Coeffs>
  requires (sizeof...(Coeffs) == Degree + 1)
#else
  template <typename... Coeffs, enable_if_arithmetic<Coeffs...> = true,
            enable_if<(sizeof...(Coeffs) == Degree + 1)> = true>
#endif
  constexpr polynomial(Coeffs... coeffs)
      : m_coefficients{static_cast<Real>(coeffs)...} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
#endif
  constexpr polynomial(tensor<OtherReal, Degree + 1> const& coeffs)
      : m_coefficients{make_array<Real>(coeffs.data())} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
#endif
  constexpr polynomial(std::array<OtherReal, Degree + 1> const& coeffs)
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
  auto c(size_t i) const -> auto const& { return m_coefficients[i]; }
  auto c(size_t i) -> auto& { return m_coefficients[i]; }
  //----------------------------------------------------------------------------
  auto coefficient(size_t i) const -> auto const& { return m_coefficients[i]; }
  auto coefficient(size_t i) -> auto const& { return m_coefficients[i]; }
  //----------------------------------------------------------------------------
  auto coefficients() const -> auto const& { return m_coefficients; }
  auto coefficients() -> auto const& { return m_coefficients; }
  //----------------------------------------------------------------------------
  constexpr auto set_coefficients(
      std::array<Real, Degree + 1> const& coeffs) -> void{
    m_coefficients = coeffs;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_coefficients(std::array<Real, Degree + 1>&& coeffs) -> void {
    m_coefficients = std::move(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if_arithmetic<OtherReal> = true>
#endif
  constexpr auto set_coefficients(
      std::array<OtherReal, Degree + 1> const& coeffs) -> void {
    m_coefficients = make_array<Real>(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic... Coeffs>
  requires (sizeof...(Coeffs) == Degree + 1)
#else
  template <typename... Coeffs, enable_if_arithmetic<Coeffs...> = true,
            enable_if<(sizeof...(Coeffs) == Degree + 1)> = true>
#endif
  constexpr auto set_coefficients(Coeffs... coeffs) -> void {
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
  auto print(std::ostream& out, std::string const& x) const -> std::ostream& {
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
    -> polynomial<common_type<Coeffs...>, sizeof...(Coeffs) - 1>;
template <typename Real, size_t N>
polynomial(tensor<Real, N> const&) -> polynomial<Real, N - 1>;
//------------------------------------------------------------------------------
// diff
//------------------------------------------------------------------------------
template <typename Real, size_t Degree>
constexpr auto diff(polynomial<Real, Degree> const& f) {
  return f.diff();
}
//------------------------------------------------------------------------------
// solve
//------------------------------------------------------------------------------
/// solve a + bx
template <typename Real>
auto solve(polynomial<Real, 1> const& p) -> std::vector<Real> {
  if (p.c(1) == 0) { return {}; }
  return {-p.c(0) / p.c(1)};
}
//------------------------------------------------------------------------------
/// solve a + bx + cxx
template <typename Real>
auto solve(polynomial<Real, 2> const& p) -> std::vector<Real> {
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
  Real q = (b > 0) ? -0.5 * (b + std::sqrt(discr)) : -0.5 * (b - std::sqrt(discr));
  solutions.push_back(q / c);
  solutions.push_back(a / q);
  std::swap(solutions[0], solutions[1]);
  return solutions;
}
//------------------------------------------------------------------------------
// I/O
//------------------------------------------------------------------------------
template <typename Real, size_t Degree>
auto& operator<<(std::ostream& out, polynomial<Real, Degree> const& f) {
  return f.print(out, "x");
}
//------------------------------------------------------------------------------
// type_traits
//------------------------------------------------------------------------------
template <typename T>
struct is_polynomial_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t Degree>
struct is_polynomial_impl<polynomial<Real, Degree>> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr bool is_polynomial = is_polynomial_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ... Ts>
using enable_if_polynomial = enable_if<is_polynomial<Ts>...>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
