#ifndef TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SPLIT_BEHAVIOR_H
#define TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SPLIT_BEHAVIOR_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/math.h>
#include <array>
//==============================================================================
namespace tatooine::detail::autonomous_particle {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct split_behaviors;
//==============================================================================
template <floating_point Real>
struct split_behaviors<Real, 2> {
  static auto constexpr one            = Real(1);
  static auto constexpr half           = one / Real(2);
  static auto constexpr quarter        = one / Real(4);
  static auto constexpr three_quarters = 3 * quarter;
  static auto constexpr sqrt5          = gcem::sqrt<Real>(5);
  using vec_t                          = vec<Real, 2>;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct two_splits {
    static auto constexpr sqrt2    = gcem::sqrt(real_t(2));
    static auto constexpr sqr_cond = 2;
    static auto constexpr half     = Real(1) / Real(2);
    static auto constexpr quarter  = Real(1) / Real(4);
    static constexpr auto radii =
        std::array{vec_t{1 / sqrt2, half}, vec_t{1 / sqrt2, half}};
    static constexpr auto offsets = std::array{vec_t{0, -half}, vec_t{0, half}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct three_splits {
    static auto constexpr sqr_cond = Real(4);
    static constexpr auto radii    = std::array{
        vec_t{half, quarter}, vec_t{one, half}, vec_t{half, quarter}};
    static constexpr auto offsets = std::array{
        vec_t{0, -three_quarters}, vec_t{0, 0}, vec_t{0, three_quarters}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct five_splits {
    static auto constexpr sqr_cond = Real(6 + sqrt5 * 2);
    static auto constexpr radii    = std::array{
        vec_t{1, 1 / (sqrt5 + 1)}, vec_t{1, (sqrt5 + 3) / (sqrt5 * 2 + 2)},
        vec_t{1, 1}, vec_t{1, (sqrt5 + 3) / (sqrt5 * 2 + 2)},
        vec_t{1, 1 / (sqrt5 + 1)}};
    static auto constexpr offsets = std::array{
        vec_t{0, 0}, vec_t{0, 0}, vec_t{0, 0}, vec_t{0, 0}, vec_t{0, 0}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct seven_splits {
    static auto constexpr sqr_cond = 4.493959210 * 4.493959210;
    static auto constexpr radii    = std::array{
        real_t(.9009688678), real_t(.6234898004), real_t(.2225209338)};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  struct centered_four {
    static auto constexpr x5       = Real(0.4830517593887872);
    static auto constexpr sqr_cond = Real{4};
    static auto constexpr radii =
        std::array{vec_t{x5, x5 / 2},
                   vec_t{x5, x5 / 2},
                   vec_t{x5, x5 / 2},
                   vec_t{x5, x5 / 2},
                   vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)},
                   vec_t{real_t(1) / real_t(2), real_t(1) / real_t(4)}};
    static auto constexpr offsets = std::array{
        vec_t{-x5, -x5 / 2}, vec_t{x5, -x5 / 2},      vec_t{-x5, x5 / 2},
        vec_t{x5, x5 / 2},   vec_t{0, real_t(3) / 4}, vec_t{0, -real_t(3) / 4}};
  };
};
//==============================================================================
template <floating_point Real>
struct split_behaviors<Real, 3> {
  static auto constexpr half           = 1 / Real(2);
  static auto constexpr quarter        = 1 / Real(4);
  static auto constexpr three_quarters = 3 * quarter;
  using vec_t                          = vec<Real, 3>;
  struct three_splits {
    static auto constexpr sqr_cond = Real{4};
    static constexpr auto radii =
        std::array{vec_t{half, half, quarter}, vec_t{1, 1, half},
                   vec_t{half, half, quarter}};
    static constexpr auto offsets =
        std::array{vec_t{0, 0, -three_quarters}, vec_t{0, 0, 0},
                   vec_t{0, 0, three_quarters}};
  };
};
//==============================================================================
}  // namespace tatooine::detail::autonomous_particle
//==============================================================================
#endif
