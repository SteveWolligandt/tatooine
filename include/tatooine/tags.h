#ifndef TATOOINE_TAGS_H
#define TATOOINE_TAGS_H
//==============================================================================
namespace tatooine::tag {
//==============================================================================
struct parallel_t {};
static constexpr parallel_t parallel;
struct sequential_t {};
static constexpr sequential_t sequential;
struct frobenius_t {};
static constexpr frobenius_t frobenius;
struct full_t {};
static constexpr full_t full;
struct economy_t {};
static constexpr economy_t economy;
struct eye_t {};
static constexpr eye_t eye;
struct automatic_t {};
static constexpr automatic_t automatic;
struct forward_t {};
static constexpr forward_t forward;
struct backward_t {};
static constexpr backward_t backward;
struct central_t {};
static constexpr central_t central;
struct quadratic_t {};
static constexpr quadratic_t quadratic;
struct analytical_t {};
static constexpr analytical_t analytical;
struct numerical_t {};
static constexpr numerical_t numerical;
struct heap {};
struct stack {};

template <typename Real>
struct fill {
  Real value;
};
template <typename Real>
fill(Real)->fill<Real>;

struct zeros_t {};
static constexpr zeros_t zeros;

struct ones_t {};
static constexpr ones_t ones;
//==============================================================================
}  // namespace tatooine::tags
//==============================================================================
#endif
