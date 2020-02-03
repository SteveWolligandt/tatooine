#ifndef TATOOINE_INTERPOLATION_H
#define TATOOINE_INTERPOLATION_H

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

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
   hermite(const Data& fx0,   const Data& fx1,
           const Data& fx0dx, const Data& fx1dx)
       : m_fx0{fx0}, m_fx1{fx1}, m_fx0dx{fx0dx}, m_fx1dx{fx1dx} {}

   //----------------------------------------------------------------------------
   // methods
   //----------------------------------------------------------------------------
   /// actual calculation of hermite interpolation
   static constexpr auto interpolate(const Data& A, const Data& B,
                                     const Data& C, const Data& D,
                                     Real t) noexcept {
     auto a = B;
     auto b = (C - A) * 0.5;
     auto c = -(D - 4.0 * C + (5.0 * B) - 2.0 * A) * 0.5;
     auto d = (D - (3.0 * C) + (3.0 * B) - A) * 0.5;
     return a + b * t + c * t * t + d * t * t * t;
  }

  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys
  template <typename Iter>
  static constexpr auto interpolate_iter(Iter A, Iter B, Iter begin, Iter end,
                                         Real t) {
    if (A == B) { return *A; }
    if (t == 0) { return *A; }
    if (t == 1) { return *B; }
    const auto left  = *A;
    const auto right = *B;

    const auto pre_left =
        A == begin ? 3 * left - 3 * right + *next(B) : *prev(A);
    const auto post_right =
        B == prev(end) ? 3 * right - 3 * left + *prev(A) : *next(B);
    return interpolate(pre_left, left, right, post_right, t);
  }

  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys for multidimensional interpolations
  template <typename Iter, typename... Xs>
  static constexpr auto interpolate_iter(Iter A, Iter B, Iter begin, Iter end,
                                         Real t, Real x, Xs&&... xs) {
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
    return interpolate(pre_left, left, right, post_right, t);
  }
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
