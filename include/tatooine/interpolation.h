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
template <typename Real, typename Derived>
struct base1 {
  /// actual nearest_neighbour interpolation
  template <typename Data>
  static constexpr auto interpolate(const Data& A, const Data& B,
                                    Real t) noexcept {
    return Derived::interpolate(A, B, t);
  }

  /// nearest_neighbour interpolation using iterators
  template <typename Iter>
  static constexpr auto interpolate_iter(Iter A, Iter B, Real t) {
    return interpolate(*A, *B, t);
  }

  /// nearest_neighbour interpolation using iterators for multi-dimensional
  /// fields
  template <typename Iter, typename... Xs>
  static constexpr auto interpolate_iter(Iter A, Iter B, Real t, Real x,
                                         Xs&&... xs) {
    return interpolate((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);
  }

  /// nearest_neighbour interpolation using iterators needed for regular_grid
  template <typename Iter>
  static constexpr auto interpolate_iter(Iter A, Iter         B, Iter /*begin*/,
                                         Iter /*end*/, Real t) {
    return interpolate(*A, *B, t);
  }

  /// nearest_neighbour interpolation using iterators for multi-dimensional
  /// fields. needed for regular_grid
  template <typename Iter, typename... Xs>
  static constexpr auto interpolate_iter(Iter A, Iter         B, Iter /*begin*/,
                                         Iter /*end*/, Real t, Real x,
                                         Xs&&... xs) {
    return interpolate((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);
  }
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
  using cubic_polynomial                       = polynomial<Real, 3>;
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
  cubic_polynomial m_polynomial;

   //----------------------------------------------------------------------------
   // ctors
   //----------------------------------------------------------------------------
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
   const auto& polynomial() const { return m_polynomial; }
   auto&       polynomial() { return m_polynomial; }
};
//==============================================================================
template <typename Real, size_t N>
struct hermite<vec<Real, N>> {
  //----------------------------------------------------------------------------
  // traits
  //----------------------------------------------------------------------------
 public:
  static constexpr bool needs_first_derivative = true;
  using real_t                                 = Real;
  using vec_t                                  = vec<Real, N>;
  using cubic_polynomial                       = polynomial<Real, 3>;
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
  std::array<cubic_polynomial, N> m_polynomials;
  //----------------------------------------------------------------------------
  // ctors
  //----------------------------------------------------------------------------
  constexpr hermite(const vec_t& fx0, const vec_t& fx1, const vec_t& fx0dx,
                    const vec_t& fx1dx)
      : m_polynomials{
            make_array<cubic_polynomial, N>(cubic_polynomial{0, 0, 0, 0})} {
    mat<Real, 4, N> B;
    for (size_t i = 0; i < N; ++i) {
      B(0, i) = fx0(i);
      B(1, i) = fx1(i);
      B(2, i) = fx0dx(i);
      B(3, i) = fx1dx(i);
    }
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Real>
struct linear : base1<Real, linear<Real>> {
  /// actual linear interpolation
  template <typename Data>
  static constexpr auto interpolate(const Data& A, const Data& B,
                                    Real t) noexcept {
    return (1 - t) * A + t * B;
  }

  template <typename Data>
  static constexpr auto interpolate(const Data& A, const Data& B,
                                    const Data& C, Real a, Real b,
                                    Real c) noexcept {
    if (auto sum = a + b + c; std::abs(sum - 1) > 1e-6) {
      std::cout << a << " + " << b << " + " << c << " = " << sum << '\n';
      assert(false && "barycentric coordinates do not sum up to 1");
    }
    return A * a + B * b + C * c;
  }
};  // struct linear

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename Real>
struct nearest_neighbour : base1<Real, nearest_neighbour<Real>> {
  /// actual nearest_neighbour interpolation
  template <typename Data>
  static constexpr auto interpolate(const Data& A, const Data& B,
                                    Real t) noexcept {
    return t < 0.5 ? A : B;
    // return (1 - t) * A + t * B;
  }
};  // struct nearest_neighbour

//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================

#endif
