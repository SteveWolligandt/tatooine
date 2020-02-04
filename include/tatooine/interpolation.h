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
  // members
  //----------------------------------------------------------------------------
 public:
  polynomial_t m_polynomial;

  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear()                    = default;
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
struct linear<vec<Real, N>> {
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
  // members
  //----------------------------------------------------------------------------
 public:
  std::array<polynomial_t, N> m_polynomials;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr linear() : m_polynomials{make_array<polynomial_t, N>()} {}
  constexpr linear(const linear&)            = default;
  constexpr linear(linear&&)                 = default;
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
struct hermite<vec<Real, N>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = true;
  using real_t                                 = Real;
  using vec_t                                  = vec<Real, N>;
  using polynomial_t                       = polynomial<Real, 3>;
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

//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================

#endif
