#ifndef TATOOINE_INTERPOLATION_H
#define TATOOINE_INTERPOLATION_H

//==============================================================================
namespace tatooine::interpolation {
//==============================================================================

template <typename real_t>
struct hermite {
  //! actual calculation of hermite interpolation
  static constexpr auto interpolate(real_t A, real_t B, real_t C, real_t D,
                                    real_t t) noexcept {
    real_t a = B;
    real_t b = (C - A) * 0.5;
    real_t c = -(D - 4.0 * C + (5.0 * B) - 2.0 * A) * 0.5;
    real_t d = (D - (3.0 * C) + (3.0 * B) - A) * 0.5;
    return a + b * t + c * t * t + d * t * t * t;
  }

  //! hermite interpolation using iterators and border treatment from
  //! "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  //! Keys
  template <typename iter>
  static constexpr auto interpolate(iter A, iter B, iter begin, iter end,
                                    real_t t) {
    const auto left  = *A;
    const auto right = *B;

    const auto pre_left =
        A == begin ? 3 * left - 3 * right + *next(B) : *prev(A);
    const auto post_right =
        B == prev(end) ? 3 * right - 3 * left + *prev(A) : *next(B);
    return interpolate(pre_left, left, right, post_right, t);
  }

  //! hermite interpolation using iterators and border treatment from
  //! "Cubic Convolution Interpolation for Digital Image Processing" Robert G.
  //! Keys for multidimensional interpolations
  template <typename iter, typename... Xs>
  static constexpr auto interpolate(iter A, iter B, iter begin, iter end,
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
struct linear {
  //! actual linear interpolation
  static constexpr auto interpolate(real_t A, real_t B, real_t t) noexcept {
    return (1 - t) * A + t * B;
  }

  //! linear interpolation using iterators
  template <typename iter>
  static constexpr auto interpolate(iter A, iter B, real_t t) {
    return interpolate(*A, *B, t);
  }

  //! linear interpolation using iterators for multi-dimensional fields
  template <typename iter, typename... Xs>
  static constexpr auto interpolate(iter A, iter B, real_t t, real_t x,
                                    Xs&&... xs) {
    return interpolate((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);
  }

  //! linear interpolation using iterators needed for regular_grid
  template <typename iter>
  static constexpr auto interpolate(iter A, iter         B, iter /*begin*/,
                                    iter /*end*/, real_t t) {
    return interpolate(*A, *B, t);
  }

  //! linear interpolation using iterators for multi-dimensional fields. needed
  //! for regular_grid
  template <typename iter, typename... Xs>
  static constexpr auto interpolate(iter A, iter         B, iter /*begin*/,
                                    iter /*end*/, real_t t, real_t x,
                                    Xs&&... xs) {
    return interpolate((*A).sample(x, std::forward<Xs>(xs)...),
                       (*B).sample(x, std::forward<Xs>(xs)...), t);
  }
};  // class linear

//==============================================================================
}  // namespace tatooine::interpolation
//==============================================================================

#endif
