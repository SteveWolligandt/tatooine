#ifndef TATOOINE_INTERPOLATION_H
#define TATOOINE_INTERPOLATION_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

#include "polynomial.h"
#include "tensor.h"

//==============================================================================
namespace tatooine::interpolation {
//==============================================================================
template <typename Real>
struct linear {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = false;
  using real_t                                 = Real;
  using polynomial_t                       = polynomial<Real, 1>;
  static constexpr size_t num_dimensions() { return 1; }
  static_assert(std::is_arithmetic_v<Real>);

  //----------------------------------------------------------------------------
  // static members
  //----------------------------------------------------------------------------
 public:
  static constexpr mat<Real, 2, 2> A{{ 1,  0},
                                     {-1,  1}};
  //----------------------------------------------------------------------------
  // factories
  //----------------------------------------------------------------------------
  constexpr static Real interpolate_via_2_values(Real a, Real b, Real t) {
    return a * (1-t) + b * t;
  }
  //----------------------------------------------------------------------------
  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys
  template <typename Iterator>
  static constexpr Real from_iterators(Iterator A, Iterator B, Real t) {
    return interpolate_via_2_values(*A, *B, t);
  }
  //----------------------------------------------------------------------------
  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys for multidimensional interpolations
  template <typename Iterator, typename... Xs>
  static constexpr Real from_iterators(Iterator A, Iterator B, Real t, Real x,
                                       Xs&&... xs) {
    return interpolate_via_2_values((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);

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
  constexpr linear(const Real& fx0, const Real& fx1)
      : m_polynomial{0, 0, 0, 0} {
    vec<Real, 2> b{fx0, fx1};
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
  using polynomial_t                           = polynomial<Real, 1>;
  static constexpr size_t num_dimensions() { return N; }

  //----------------------------------------------------------------------------
  // static members
  //----------------------------------------------------------------------------
 public:
  static constexpr mat<Real, 2, 2> A{{ 1,  0},
                                     {-1,  1}};
  //----------------------------------------------------------------------------
  // factories
  //----------------------------------------------------------------------------
  constexpr static vec_t interpolate_via_2_values(const vec_t& a,
                                                  const vec_t& b, Real t) {
    return a * (1 - t) + b * t;
  }
  //----------------------------------------------------------------------------
  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys
  template <typename Iterator>
  static constexpr vec_t from_iterators(Iterator A, Iterator B, Real t) {
    return interpolate_via_2_values(*A, *B, t);
  }
  //----------------------------------------------------------------------------
  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys for multidimensional interpolations
  template <typename Iterator, typename... Xs>
  static constexpr vec_t from_iterators(Iterator A, Iterator B, Real t, Real x,
                                       Xs&&... xs) {
    return interpolate_via_2_values((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);

  }

  //----------------------------------------------------------------------------
  // members
  //----------------------------------------------------------------------------
 public:
  std::array<polynomial_t, N> m_polynomials;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear() : m_polynomials{make_array<polynomial_t, N>()} {}
  constexpr linear(const linear&) = default;
  constexpr linear(linear&&)      = default;
  constexpr linear& operator=(const linear&) = default;
  constexpr linear& operator=(linear&&)      = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr linear(const vec_t& fx0, const vec_t& fx1)
      : m_polynomials{make_array<polynomial_t, N>(polynomial_t{0, 0})} {
    mat<Real, 2, N> B;
    B.row(0) = fx0;
    B.row(1) = fx1;
    auto C   = A * B;
    for (size_t i = 0; i < N; ++i) {
      m_polynomials[i].set_coefficients(C(0, i), C(1, i));
    }
  }

  //----------------------------------------------------------------------------
  // methods
  //----------------------------------------------------------------------------
  template <size_t... Is>
  constexpr auto evaluate(Real t, std::index_sequence<Is...>) const {
    return vec_t{m_polynomials[Is](t)...};
  }
  constexpr auto evaluate(Real t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  const auto& polynomial(size_t i) const { return m_polynomials[i]; }
  auto&       polynomial(size_t i) { return m_polynomials[i]; }
  //----------------------------------------------------------------------------
  const auto& polynomials() const { return m_polynomials; }
  auto&       polynomials() { return m_polynomials; }
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
  using polynomial_t                       = polynomial<Real, 3>;
  static constexpr size_t num_dimensions() { return 1; }
  static_assert(std::is_arithmetic_v<Real>);

  //----------------------------------------------------------------------------
  // static members
  //----------------------------------------------------------------------------
 public:
  static constexpr mat<Real, 4, 4> A{{ 1,  0,  0,  0},
                                     { 0,  0,  1,  0},
                                     {-3,  3, -2, -1},
                                     { 2, -2,  1,  1}};
  //----------------------------------------------------------------------------
  // factories
  //----------------------------------------------------------------------------
  static constexpr Real interpolate_via_4_values(const Real& A, const Real& B,
                                                 const Real& C, const Real& D,
                                                 Real t) noexcept {
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
  static constexpr Real from_iterators(Iterator A, Iterator B, Iterator begin,
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
  static constexpr Real from_iterators(Iterator A, Iterator B, Iterator begin,
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
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_t m_polynomial;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr hermite()                          = default;
  constexpr hermite(const hermite&)            = default;
  constexpr hermite(hermite&&)                 = default;
  constexpr hermite& operator=(const hermite&) = default;
  constexpr hermite& operator=(hermite&&)      = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr hermite(const Real& fx0, const Real& fx1, const Real& fx0dx,
                    const Real& fx1dx)
      : m_polynomial{0, 0, 0, 0} {
    vec<Real, 4> b{fx0, fx1, fx0dx, fx1dx};
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
  using polynomial_t                           = polynomial<Real, 3>;
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
  std::array<polynomial_t, N> m_polynomials;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr hermite() : m_polynomials{make_array<polynomial_t, N>()} {}
  constexpr hermite(const hermite&)            = default;
  constexpr hermite(hermite&&)                 = default;
  constexpr hermite& operator=(const hermite&) = default;
  constexpr hermite& operator=(hermite&&)      = default;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  constexpr hermite(const vec_t& fx0, const vec_t& fx1, const vec_t& fx0dx,
                    const vec_t& fx1dx)
      : m_polynomials{
            make_array<polynomial_t, N>(polynomial_t{0, 0, 0, 0})} {
    mat<Real, 4, N> B;
    B.row(0) = fx0;
    B.row(1) = fx1;
    B.row(2) = fx0dx;
    B.row(3) = fx1dx;
    auto C = A * B;
    for (size_t i = 0; i < N; ++i) {
      m_polynomials[i].set_coefficients(C(0, i), C(1, i), C(2, i), C(3, i));
    }
  }

  //----------------------------------------------------------------------------
  // factories
  //----------------------------------------------------------------------------
  static constexpr vec_t interpolate_via_4_values(const vec_t& A, const vec_t& B,
                                                 const vec_t& C, const vec_t& D,
                                                 Real t) noexcept {
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
  template <size_t... Is>
  constexpr auto evaluate(Real t, std::index_sequence<Is...>) const {
    return vec_t{m_polynomials[Is](t)...};
  }
  constexpr auto evaluate(Real t) const {
    return evaluate(t, std::make_index_sequence<N>{});
  }
  constexpr auto operator()(Real t) const { return evaluate(t); }
  //----------------------------------------------------------------------------
  const auto& polynomial(size_t i) const { return m_polynomials[i]; }
  auto&       polynomial(size_t i) { return m_polynomials[i]; }
  //----------------------------------------------------------------------------
  const auto& polynomials() const { return m_polynomials; }
  auto&       polynomials() { return m_polynomials; }
};
template <typename Real, size_t N>
struct hermite<vec<Real, N>> : hermite<tensor<Real, N>> {
  using hermite<tensor<Real, N>>::hermite;
};

//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================

#endif
