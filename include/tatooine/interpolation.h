#ifndef __TATOOINE_INTERPOLATION_H__
#define __TATOOINE_INTERPOLATION_H__

#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

//==============================================================================
namespace tatooine::interpolation {
//==============================================================================

template <typename real_t, typename derived_t>
struct base1 {
  /// actual nearest_neighbour interpolation
  template <typename data_t>
  static constexpr auto interpolate(const data_t& A, const data_t& B,
                                    real_t t) noexcept {
    return derived_t::interpolate(A, B, t);
  }

  /// nearest_neighbour interpolation using iterators
  template <typename iter>
  static constexpr auto interpolate_iter(iter A, iter B, real_t t) {
    return interpolate(*A, *B, t);
  }

  /// nearest_neighbour interpolation using iterators for multi-dimensional
  /// fields
  template <typename iter, typename... Xs>
  static constexpr auto interpolate_iter(iter A, iter B, real_t t, real_t x,
                                         Xs&&... xs) {
    return interpolate((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);
  }

  /// nearest_neighbour interpolation using iterators needed for regular_grid
  template <typename iter>
  static constexpr auto interpolate_iter(iter A, iter         B, iter /*begin*/,
                                         iter /*end*/, real_t t) {
    return interpolate(*A, *B, t);
  }

  /// nearest_neighbour interpolation using iterators for multi-dimensional
  /// fields. needed for regular_grid
  template <typename iter, typename... Xs>
  static constexpr auto interpolate_iter(iter A, iter         B, iter /*begin*/,
                                         iter /*end*/, real_t t, real_t x,
                                         Xs&&... xs) {
    return interpolate((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);
  }
};

template <typename real_t>
struct hermite {
  /// actual calculation of hermite interpolation
  template <typename data_t>
  static constexpr auto interpolate(const data_t& A, const data_t& B,
                                    const data_t& C, const data_t& D,
                                    real_t t) noexcept {
    auto a = B;
    auto b = (C - A) * 0.5;
    auto c = -(D - 4.0 * C + (5.0 * B) - 2.0 * A) * 0.5;
    auto d = (D - (3.0 * C) + (3.0 * B) - A) * 0.5;
    return a + b * t + c * t * t + d * t * t * t;
  }

  /// hermite interpolation using iterators and border treatment from
  /// "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  /// Keys
  template <typename iter>
  static constexpr auto interpolate_iter(iter A, iter B, iter begin, iter end,
                                         real_t t) {
    if (A == B) {
      return *A;
    }
    if (t == 0) {
      return *A;
    }
    if (t == 1) {
      return *B;
    }
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
  template <typename iter, typename... Xs>
  static constexpr auto interpolate_iter(iter A, iter B, iter begin, iter end,
                                         real_t t, real_t x, Xs&&... xs) {
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

template <typename real_t>
struct linear : base1<real_t, linear<real_t>> {
  /// actual linear interpolation
  template <typename data_t>
  static constexpr auto interpolate(const data_t& A, const data_t& B,
                                    real_t t) noexcept {
    return (1 - t) * A + t * B;
  }

  template <typename data_t>
  static constexpr auto interpolate(const data_t& A, const data_t& B,
                                    const data_t& C, real_t a, real_t b,
                                    real_t c) noexcept {
    if (auto sum = a + b + c; std::abs(sum - 1) > 1e-6) {
      std::cout << a << " + " << b << " + " << c << " = " << sum << '\n';
      assert(false && "barycentric coordinates do not sum up to 1");
    }
    return A * a + B * b + C * c;
  }
};  // struct linear

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename real_t>
struct nearest_neighbour : base1<real_t, nearest_neighbour<real_t>> {
  /// actual nearest_neighbour interpolation
  template <typename data_t>
  static constexpr auto interpolate(const data_t& A, const data_t& B,
                                    real_t t) noexcept {
    return t < 0.5 ? A : B;
    // return (1 - t) * A + t * B;
  }
};  // struct nearest_neighbour

//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================

#endif
