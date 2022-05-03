#ifndef TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SPLIT_BEHAVIOR_H
#define TATOOINE_DETAIL_AUTONOMOUS_PARTICLE_SPLIT_BEHAVIOR_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/math.h>
#include <tatooine/vec.h>

#include <array>
//==============================================================================
namespace tatooine::detail::autonomous_particle {
//==============================================================================
template <floating_point Real, std::size_t NumDimensions>
struct split_behaviors;
//==============================================================================
/// See \ref autonomous_particle_page_split_behaviors_2d "Split Behavior in 2D".
template <floating_point Real>
struct split_behaviors<Real, 2> {
  static auto constexpr one            = Real(1);
  static auto constexpr half           = one / Real(2);
  static auto constexpr third          = one / Real(3);
  static auto constexpr quarter        = one / Real(4);
  static auto constexpr sixth          = one / Real(6);
  static auto constexpr three_quarters = 3 * quarter;
  static auto constexpr two_thirds     = 2 * third;
  static auto constexpr three_sixths   = 3 * sixth;
  static auto constexpr sqrt2          = gcem::sqrt<Real>(2);
  static auto constexpr sqrt5          = gcem::sqrt<Real>(5);
  using vec_t                          = vec<Real, 2>;
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \image html autonomous_particle/splits/2d/2splits.svg
  struct two_splits {
    static auto constexpr split_cond = sqrt2;
    static constexpr auto radii =
        std::array{vec_t{1 / sqrt2, 1 / sqrt2 / sqrt2},
                   vec_t{1 / sqrt2, 1 / sqrt2 / sqrt2}};
    static constexpr auto offsets =
        std::array{vec_t{0, sqrt2 / 2 / sqrt2}, vec_t{0, -sqrt2 / 2 / sqrt2}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \image html autonomous_particle/splits/2d/3splits.svg
  struct three_splits {
    static auto constexpr split_cond = Real(2);
    static constexpr auto radii      = std::array{
        vec_t{half, quarter}, vec_t{one, half}, vec_t{half, quarter}};
    static constexpr auto offsets = std::array{
        vec_t{0, -three_quarters}, vec_t{0, 0}, vec_t{0, three_quarters}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \image html autonomous_particle/splits/2d/3splits_alternative.svg
  struct three_in_square_splits {
    static auto constexpr split_cond = Real(3);
    static constexpr auto radii =
        std::array{vec_t{one, third}, vec_t{one, third}, vec_t{one, third}};
    static constexpr auto offsets =
        std::array{vec_t{0, -two_thirds}, vec_t{0, 0}, vec_t{0, two_thirds}};
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \image html autonomous_particle/splits/2d/5splits.svg
  struct five_splits {
    static auto constexpr r1 = one + sqrt5;  // larger radius when splitting
    static auto constexpr r2 = one;          // smaller radius when splitting
    static auto constexpr r3 = one / (one + sqrt5);  // outer radius
    static auto constexpr x3 = r1 - r3;              // outer offset
    static auto constexpr r4 =
        half * sqrt5 - one / (one + sqrt5);  // middle radius
    static auto constexpr x4 =
        one + half * sqrt5 - one / (one + sqrt5);  // middle offset

    static auto constexpr split_cond = r1;
    static auto constexpr radii      = std::array{
        vec_t{r3, r3 / r1}, vec_t{r4, r4 / r1}, vec_t{r2, r2 / r1},
        vec_t{r4, r4 / r1}, vec_t{r3, r3 / r1},
    };
    static auto constexpr offsets = std::array{
        vec_t{0, x3 / r1},  vec_t{0, x4 / r1},  vec_t{0, 0},
        vec_t{0, -x4 / r1}, vec_t{0, -x3 / r1},
    };
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// \image html autonomous_particle/splits/2d/7splits_alternative.svg
  struct three_and_four_splits {
    static auto constexpr r01 = Real(2);
    static auto constexpr r02 = Real(1);

    static auto constexpr x1 = Real(0);
    static auto constexpr y1 = Real(0);
    static auto constexpr r1 = Real(1);

    static auto constexpr x2 = 3 * half;
    static auto constexpr y2 = Real(0);
    static auto constexpr r2 = half;

    static auto constexpr x4 = Real(1.077350269);
    static auto constexpr y4 = Real(0.5977169814);
    static auto constexpr r4 = Real(0.2320508081);

    static auto constexpr split_cond = r01 / r02;

    static auto constexpr radii = std::array{
        vec_t{r1, r1 / r01}, vec_t{r2, r2 / r01}, vec_t{r2, r2 / r01},
        vec_t{r4, r4 / r01}, vec_t{r4, r4 / r01}, vec_t{r4, r4 / r01},
        vec_t{r4, r4 / r01},
    };
    static auto constexpr offsets = std::array{
        vec_t{0, 0},           vec_t{y2, x2 / r01},  vec_t{y2, -x2 / r01},
        vec_t{y4, x4 / r01},   vec_t{y4, -x4 / r01}, vec_t{-y4, x4 / r01},
        vec_t{-y4, -x4 / r01},
    };
  };
  //============================================================================
  /// \image html autonomous_particle/splits/2d/7splits.svg
  struct seven_splits {
    static auto constexpr rr  = Real(4.493959210);
    static auto constexpr rr1 = Real(0.9009688678);
    static auto constexpr rr2 = Real(0.6234898004);

    static auto constexpr split_cond = rr;

    static auto constexpr radii = std::array{
        vec_t{1, 1 / rr},
        vec_t{(1 / rr), (1 / rr) / rr},
        vec_t{(1 / rr), (1 / rr) / rr},
        vec_t{rr1, rr1 / rr},
        vec_t{rr1, rr1 / rr},
        vec_t{rr2, rr2 / rr},
        vec_t{rr2, rr2 / rr},
    };
    static auto constexpr offsets = std::array{
        vec_t{0, 0},
        vec_t{0, (1 + 2 * rr1 + 2 * rr2 + (1 / rr)) / rr},
        vec_t{0, (-1 - 2 * rr1 - 2 * rr2 - (1 / rr)) / rr},
        vec_t{0, (1 + rr1) / rr},
        vec_t{0, (-1 - rr1) / rr},
        vec_t{0, (1 + 2 * rr1 + rr2) / rr},
        vec_t{0, (-1 - 2 * rr1 - rr2) / rr},
    };
  };
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // struct centered_four {
  //  static auto constexpr x5       = Real(0.4830517593887872);
  //  static auto constexpr split_cond = Real(2);
  //  static auto constexpr radii =
  //      std::array{vec_t{x5, x5 / 2},
  //                 vec_t{x5, x5 / 2},
  //                 vec_t{x5, x5 / 2},
  //                 vec_t{x5, x5 / 2},
  //                 vec_t{real_type(1) / real_type(2), real_type(1) / real_type(4)},
  //                 vec_t{real_type(1) / real_type(2), real_type(1) / real_type(4)}};
  //  static auto constexpr offsets = std::array{
  //      vec_t{-x5, -x5 / 2}, vec_t{x5, -x5 / 2},      vec_t{-x5, x5 / 2},
  //      vec_t{x5, x5 / 2},   vec_t{0, real_type(3) / 4}, vec_t{0, -real_type(3) /
  //      4}};
  //};
};
//==============================================================================
/// See \ref autonomous_particle_page_split_behaviors_3d "Split Behavior in 3D".
template <floating_point Real>
struct split_behaviors<Real, 3> {
  static auto constexpr half           = 1 / Real(2);
  static auto constexpr quarter        = 1 / Real(4);
  static auto constexpr three_quarters = 3 * quarter;
  using vec_t                          = vec<Real, 3>;
  struct three_splits {
    static auto constexpr split_cond = Real(2);
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
