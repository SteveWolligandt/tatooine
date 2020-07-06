#ifndef TATOOINE_INTERPOLATION_H
#define TATOOINE_INTERPOLATION_H
//==============================================================================
#include <tatooine/concepts.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

#include "polynomial.h"
#include "polynomial_line.h"
#include "tensor.h"
//==============================================================================
namespace tatooine {
namespace interpolation {
//==============================================================================
template <typename Real>
struct linear {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = false;
  using real_t                                 = Real;
  using polynomial_t                           = tatooine::polynomial<Real, 1>;
  static constexpr size_t num_dimensions() { return 1; }
  static_assert(std::is_arithmetic<Real>::value);

  //----------------------------------------------------------------------------
  // static members
  //----------------------------------------------------------------------------
 public:
  static constexpr mat<Real, 2, 2> A{{1, 0}, {-1, 1}};
  //----------------------------------------------------------------------------
  // factories
  //----------------------------------------------------------------------------
  constexpr static Real interpolate(Real a, Real b, Real t) {
    return a * (1 - t) + b * t;
  }
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_t m_polynomial;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear()              = default;
  constexpr linear(const linear&) = default;
  constexpr linear(linear&&)      = default;
  constexpr linear& operator=(const linear&) = default;
  constexpr linear& operator=(linear&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(const Real& ft0, const Real& ft1)
      : m_polynomial{0, 0} {
    vec<Real, 2> b{ft0, ft1};
    m_polynomial.set_coefficients((A * b).data());
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(Real t) const { return m_polynomial(t); }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  constexpr const auto& polynomial() const { return m_polynomial; }
  constexpr auto&       polynomial() { return m_polynomial; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, size_t N>
struct linear<tensor<Real, N>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = false;
  using real_t                                 = Real;
  using vec_t                                  = vec<Real, N>;
  using polynomial_line_t                      = polynomial_line<Real, N, 1>;
  static constexpr size_t num_dimensions() { return N; }

  //----------------------------------------------------------------------------
  // static members
  //----------------------------------------------------------------------------
 public:
  static constexpr mat<Real, 2, 2> A{{1, 0}, {-1, 1}};
  //----------------------------------------------------------------------------
  // factories
  //----------------------------------------------------------------------------
  constexpr static vec_t interpolate(const vec_t& a, const vec_t& b, Real t) {
    return a * (1 - t) + b * t;
  }
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_line_t m_curve;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear()              = default;
  constexpr linear(const linear&) = default;
  constexpr linear(linear&&)      = default;
  constexpr linear& operator=(const linear&) = default;
  constexpr linear& operator=(linear&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(const vec_t& ft0, const vec_t& ft1) {
    mat<Real, 2, N> B;
    B.row(0) = ft0;
    B.row(1) = ft1;
    auto C   = A * B;
    for (size_t i = 0; i < N; ++i) {
      m_curve.polynomial(i).set_coefficients(C(0, i), C(1, i));
    }
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(Real t) const { return m_curve(t); }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  const auto& curve() const { return m_curve; }
  auto&       curve() { return m_curve; }
};
template <typename Real, size_t N>
struct linear<vec<Real, N>> : linear<tensor<Real, N>> {
  using linear<tensor<Real, N>>::linear;
};
//==============================================================================
template <typename Real>
struct hermite {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = true;
  using real_t                                 = Real;
  using polynomial_t                           = tatooine::polynomial<Real, 3>;
  static constexpr size_t num_dimensions() { return 1; }
  static_assert(std::is_arithmetic<Real>::value);

  //----------------------------------------------------------------------------
  // static members
  //----------------------------------------------------------------------------
 public:
  static constexpr mat<Real, 4, 4> A{{ 1,  0,  0,  0},
                                     { 0,  0,  1,  0},
                                     {-3,  3, -2, -1},
                                     { 2, -2,  1,  1}};
  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_t m_polynomial;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr hermite()               = default;
  constexpr hermite(const hermite&) = default;
  constexpr hermite(hermite&&)      = default;
  constexpr hermite& operator=(const hermite&) = default;
  constexpr hermite& operator=(hermite&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr hermite(Real const t0,   Real const t1,
                    Real const ft0,  Real const ft1,
                    Real const ft0dt,Real const ft1dt)
      : m_polynomial{0, 0, 0, 0} {
    mat<Real, 4, 4> const A {{1.0,  t0, t0 * t0, t0 * t0 * t0},
                             {1.0,  t1, t1 * t1, t1 * t1 * t1},
                             {0.0, 1.0,  2 * t0,  3 * t0 * t0},
                             {0.0, 1.0,  2 * t1,  3 * t1 * t1}};
    vec<Real, 4> b{ft0, ft1, ft0dt, ft1dt};
    m_polynomial.set_coefficients(solve(A, b).data());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr hermite(const Real& ft0, const Real& ft1,
                    const Real& ft0dt, const Real& ft1dt)
      : m_polynomial{0, 0, 0, 0} {
    vec<Real, 4> b{ft0, ft1, ft0dt, ft1dt};
    m_polynomial.set_coefficients((A * b).data());
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(Real t) const { return m_polynomial(t); }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  constexpr const auto& polynomial() const { return m_polynomial; }
  constexpr auto&       polynomial() { return m_polynomial; }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Real, size_t N>
struct hermite<tensor<Real, N>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = true;
  using real_t                                 = Real;
  using vec_t                                  = vec<Real, N>;
  using polynomial_line_t                      = polynomial_line<Real, N, 3>;
  static constexpr size_t num_dimensions() { return N; }

  //----------------------------------------------------------------------------
  // static members
  //----------------------------------------------------------------------------
 public:
  static constexpr mat<Real, 4, 4> A{{ 1,  0,  0,  0},
                                     { 0,  0,  1,  0},
                                     {-3,  3, -2, -1},
                                     { 2, -2,  1,  1}};

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_line_t m_curve;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr hermite()               = default;
  constexpr hermite(const hermite&) = default;
  constexpr hermite(hermite&&)      = default;
  constexpr hermite& operator=(const hermite&) = default;
  constexpr hermite& operator=(hermite&&) = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr hermite(const vec_t& fx0,   const vec_t& fx1,
                    const vec_t& fx0dx, const vec_t& fx1dx) {
    mat<Real, 4, N> B;
    B.row(0)     = fx0;
    B.row(1)     = fx1;
    B.row(2)     = fx0dx;
    B.row(3)     = fx1dx;
    const auto C = A * B;
    for (size_t i = 0; i < N; ++i) {
      m_curve.polynomial(i).set_coefficients(
          C(0, i), C(1, i), C(2, i), C(3, i));
    }
    if (!approx_equal(m_curve(0), fx0)) {
      std::cerr << m_curve(0) << '\n';
      std::cerr << fx0 << '\n';
      throw std::runtime_error{"blub"};
    }
    if (!approx_equal(m_curve(1), fx1)) {
      std::cerr << m_curve(1) << '\n';
      std::cerr << fx1 << '\n';
      throw std::runtime_error{"blub"};
    }
    if (!approx_equal(m_curve.tangent(0), fx0dx)) {
      std::cerr << m_curve.tangent(0) << '\n';
      std::cerr << fx0dx << '\n';
      throw std::runtime_error{"blub"};
    }
    if (!approx_equal(m_curve.tangent(1), fx1dx)) {
      std::cerr << m_curve.tangent(1) << '\n';
      std::cerr << fx1dx << '\n';
      throw std::runtime_error{"blub"};
    }
  }

  //----------------------------------------------------------------------------
  // factories
  //----------------------------------------------------------------------------
  static constexpr vec_t interpolate_via_4_values(const vec_t& A,
                                                  const vec_t& B,
                                                  const vec_t& C,
                                                  const vec_t& D,
                                                  Real         t) noexcept {
    auto a = B;
    auto b = (C - A) * 0.5;
    auto c = -(D - 4.0 * C + (5.0 * B) - 2.0 * A) * 0.5;
    auto d = (D - (3.0 * C) + (3.0 * B) - A) * 0.5;
    return a + b * t + c * t * t + d * t * t * t;
  }
  //----------------------------------------------------------------------------
  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys
  template <typename Iterator>
  static constexpr vec_t from_iterators(Iterator A, Iterator B, Iterator begin,
                                        Iterator end, Real t) {
    if (A == B) { return *A; }
    if (t == 0) { return *A; }
    if (t == 1) { return *B; }
    const auto left  = *A;
    const auto right = *B;

    const auto pre_left =
        A == begin ? 3 * left - 3 * right + *next(B) : *prev(A);
    const auto post_right =
        B == prev(end) ? 3 * right - 3 * left + *prev(A) : *next(B);
    return interpolate_via_4_values(pre_left, left, right, post_right, t);
  }
  //----------------------------------------------------------------------------
  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys for multidimensional interpolations
  template <typename Iterator, typename... Xs>
  static constexpr vec_t from_iterators(Iterator A, Iterator B, Iterator begin,
                                        Iterator end, Real t, Real x,
                                        Xs&&... xs) {
    const auto left  = (*A).sample(x, std::forward<Xs>(xs)...);
    const auto right = (*B).sample(x, std::forward<Xs>(xs)...);

    const auto pre_left =
        A == begin ? 3 * left - 3 * right +
                         (*next(B)).sample(x, std::forward<Xs>(xs)...)
                   : (*prev(A)).sample(x, std::forward<Xs>(xs)...);
    const auto post_right =
        B == prev(end) ? 3 * right - 3 * left +
                             (*prev(A)).sample(x, std::forward<Xs>(xs)...)
                       : (*next(B)).sample(x, std::forward<Xs>(xs)...);
    return interpolate_via_4_values(pre_left, left, right, post_right, t);
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  constexpr auto evaluate(Real t) const { return m_curve(t); }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  const auto& curve() const { return m_curve; }
  auto&       curve() { return m_curve; }
};
template <typename Real, size_t N>
struct hermite<vec<Real, N>> : hermite<tensor<Real, N>> {
  using hermite<tensor<Real, N>>::hermite;
};

//==============================================================================
}  // namespace interpolation
}  // namespace tatooine
//==============================================================================

#endif
