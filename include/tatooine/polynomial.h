#ifndef TATOOINE_POLYNOMIAL_H
#define TATOOINE_POLYNOMIAL_H
//==============================================================================
#include <tatooine/linspace.h>
#include <tatooine/make_array.h>
#include <tatooine/tensor.h>
#include <tatooine/type_traits.h>

#include <array>
#include <ostream>
#include <type_traits>
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
  constexpr auto operator=(polynomial const& other) -> polynomial& = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto operator=(polynomial&& other) noexcept
      -> polynomial&     = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr polynomial(std::array<Real, Degree + 1> const& coeffs)
      : m_coefficients{coeffs} {}
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <typename OtherReal, size_t OtherDegree>
  requires(OtherDegree <= Degree)
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
  requires(sizeof...(Coeffs) == Degree + 1)
#else
  template <typename... Coeffs, enable_if<is_arithmetic<Coeffs...>> = true,
            enable_if<(sizeof...(Coeffs) == Degree + 1)> = true>
#endif
      constexpr polynomial(Coeffs... coeffs)
      : m_coefficients{static_cast<Real>(coeffs)...} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if<is_arithmetic<OtherReal>> = true>
#endif
  constexpr polynomial(tensor<OtherReal, Degree + 1> const& coeffs)
      : m_coefficients{make_array<Real>(coeffs.data())} {
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if<is_arithmetic<OtherReal>> = true>
#endif
  constexpr polynomial(std::array<OtherReal, Degree + 1> const& coeffs)
      : m_coefficients{make_array<Real>(coeffs)} {
  }

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
  constexpr auto set_coefficients(std::array<Real, Degree + 1> const& coeffs)
      -> void {
    m_coefficients = coeffs;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr auto set_coefficients(std::array<Real, Degree + 1>&& coeffs)
      -> void {
    m_coefficients = std::move(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic OtherReal>
#else
  template <typename OtherReal, enable_if<is_arithmetic<OtherReal>> = true>
#endif
  constexpr auto set_coefficients(
      std::array<OtherReal, Degree + 1> const& coeffs) -> void {
    m_coefficients = make_array<Real>(coeffs);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <arithmetic... Coeffs>
  requires(sizeof...(Coeffs) == Degree + 1)
#else
  template <typename... Coeffs, enable_if<is_arithmetic<Coeffs...>> = true,
            enable_if<(sizeof...(Coeffs) == Degree + 1)> = true>
#endif
      constexpr auto set_coefficients(Coeffs... coeffs) -> void {
    m_coefficients = std::array<Real, Degree + 1>{static_cast<Real>(coeffs)...};
  }
  //----------------------------------------------------------------------------
 private:
  template <size_t... Is>
  constexpr auto diff(std::index_sequence<Is...> /*seq*/) const {
    return polynomial<Real, Degree - 1>{(m_coefficients[Is + 1] * (Is + 1))...};
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
        } else if (c(1) == -1) {
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
/// solve a + b*x
template <typename Real>
auto solve(polynomial<Real, 1> const& p) -> std::vector<Real> {
  if (p.c(1) == 0) {
    return {};
  }
  return {-p.c(0) / p.c(1)};
}
//------------------------------------------------------------------------------
/// solve a + b*x + c*x^2
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
  Real q =
      (b > 0) ? -0.5 * (b + std::sqrt(discr)) : -0.5 * (b - std::sqrt(discr));
  solutions.push_back(q / c);
  solutions.push_back(a / q);
  std::swap(solutions[0], solutions[1]);
  return solutions;
}
//------------------------------------------------------------------------------
/// Solves cubic polynomial.
/// Code from here:
/// https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
template <typename Real>
auto solve(polynomial<Real, 3> const& f) {
  if (f.c(3) == 0) {
    return solve(polynomial{f.c(0), f.c(1), f.c(2)});
  }
  std::vector<Real> solutions;
  Real              sub;
  Real              A, B, C;
  Real              sq_A, p, q;
  Real              cb_p, D;

  // normal form: x^3 + Ax^2 + Bx + C = 0
  A = f.c(2) / f.c(3);
  B = f.c(1) / f.c(3);
  C = f.c(0) / f.c(3);

  //  substitute x = y - A/3 to eliminate quadric term: x^3 +px + q = 0
  sq_A = A * A;
  p    = Real(1) / 3 * (-Real(1) / 3 * sq_A + B);
  q    = Real(1) / 2 * (Real(2) / 27 * A * sq_A - Real(1) / 3 * A * B + C);

  // use Cardano's formula
  cb_p = p * p * p;
  D    = q * q + cb_p;

  constexpr Real eps     = 1e-30;
  constexpr auto is_zero = [](auto const x) {
    return ((x) > -eps && (x) < eps);
  };
  if (is_zero(D)) {
    if (is_zero(q)) {  // one triple solution
      solutions = {0};
    } else {  // one single and one double solution
      Real u    = std::cbrt(-q);
      solutions = {2 * u, -u};
    }
  } else if (D < 0) {  // Casus irreducibilis: three real solutions
    Real phi = Real(1) / 3 * std::acos(-q / std::sqrt(-cb_p));
    Real t   = 2 * std::sqrt(-p);

    solutions = {t * std::cos(phi), -t * std::cos(phi + M_PI / 3),
                 -t * std::cos(phi - M_PI / 3)};
  } else {  // one real solution
    Real sqrt_D = std::sqrt(D);
    Real u      = std::cbrt(sqrt_D - q);
    Real v      = -std::cbrt(sqrt_D + q);

    solutions = {u + v};
  }

  // resubstitute
  sub = Real(1) / 3 * A;

  for (auto& s : solutions) {
    s -= sub;
  }

  std::sort(begin(solutions), end(solutions));
  return solutions;
}
//------------------------------------------------------------------------------
/// Solves quartic polynomial.
/// Code from here:
/// https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
template <typename Real>
auto solve(polynomial<Real, 4> const& f) -> std::vector<Real> {
  if (f.c(4) == 0) {
    return solve(polynomial{f.c(0), f.c(1), f.c(2), f.c(3)});
  }

  std::vector<Real> solutions;

  // normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0
  auto const A = f.c(3) / f.c(4);
  auto const B = f.c(2) / f.c(4);
  auto const C = f.c(1) / f.c(4);
  auto const D = f.c(0) / f.c(4);

  //  substitute x = y - A/4 to eliminate cubic term: x^4 + px^2 + qx + r = 0
  auto const sq_A = A * A;
  auto const p    = -Real(3) / 8 * sq_A + B;
  auto const q    = Real(1) / 8 * sq_A * A - Real(1) / 2 * A * B + C;
  auto const r    = -Real(3) / 256 * sq_A * sq_A + Real(1) / 16 * sq_A * B -
                 Real(1) / 4 * A * C + D;

  constexpr Real eps     = 1e-30;
  constexpr auto is_zero = [](auto const x) {
    return ((x) > -eps && (x) < eps);
  };
  if (is_zero(r)) {
    // no absolute term: y(y^3 + py + q) = 0
    solutions = solve(polynomial{p, q, 0, 1});
    if (std::find(begin(solutions), end(solutions), 0) == end(solutions)) {
      solutions.push_back(0);
    }
  } else {
    // solve the resolvent cubic and take the one real solution...
    auto const z = solve(polynomial{Real(1) / 2 * r * p - Real(1) / 8 * q * q,
                                    -r, -Real(1) / 2 * p, 1})
                       .front();

    // ... to build two quadric equations
    auto u = z * z - r;
    auto v = 2 * z - p;

    if (is_zero(u)) {
      u = 0;
    } else if (u > 0) {
      u = std::sqrt(u);
    } else {
      return {};
    }

    if (is_zero(v)) {
      v = 0;
    } else if (v > 0) {
      v = std::sqrt(v);
    } else {
      return {};
    }

    solutions = solve(polynomial{z - u, q < 0 ? -v : v, 1});

    auto const other_solutions = solve(polynomial{z + u, q < 0 ? v : -v, 1});
    std::copy(begin(other_solutions), end(other_solutions),
              std::back_inserter(solutions));
  }

  // resubstitute
  auto const sub = Real(1) / 4 * A;

  for (auto& s : solutions) {
    s -= sub;
  }

  std::sort(begin(solutions), end(solutions));
  solutions.erase(std::unique(begin(solutions), end(solutions)),
                  end(solutions));
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
// template <typename ... Ts>
// using enable_if_polynomial = enable_if<is_polynomial<Ts>...>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
