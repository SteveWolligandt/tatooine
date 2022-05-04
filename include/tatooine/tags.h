#ifndef TATOOINE_TAGS_H
#define TATOOINE_TAGS_H
//==============================================================================
#include <concepts>
//==============================================================================
namespace tatooine {
//==============================================================================
struct forward_tag {};
static constexpr forward_tag forward;
struct backward_tag {};
static constexpr backward_tag backward;
template <typename T>
concept forward_or_backward_tag = (std::same_as<T, forward_tag>) ||
                                  (std::same_as<T, backward_tag>);
constexpr auto opposite(forward_tag const /*tag*/) { return backward_tag{}; }
constexpr auto opposite(backward_tag const /*tag*/) { return forward_tag{}; }
auto constexpr operator==(forward_tag const /*rhs*/,
                          backward_tag const /*rhs*/) {
  return false;
}
auto constexpr operator==(backward_tag const /*rhs*/,
                          forward_tag const /*rhs*/) {
  return false;
}
auto constexpr operator==(forward_tag const /*rhs*/,
                          forward_tag const /*rhs*/) {
  return true;
}
auto constexpr operator==(backward_tag const /*rhs*/,
                          backward_tag const /*rhs*/) {
  return true;
}
auto operator!=(forward_or_backward_tag auto const lhs,
                forward_or_backward_tag auto const rhs) {
  return !(lhs == rhs);
}
//template <typename T>
//struct is_forward_impl : std::false_type {};
//template <>
//struct is_forward_impl<forward_tag> : std::false_type {};
//template <typename T>
//static constexpr auto is_forward = is_forward_impl<T>::value;
template <typename T>
concept is_forward = std::same_as<forward_tag, std::decay_t<T>>;
template <typename T>
concept is_backward = std::same_as<backward_tag, std::decay_t<T>>;

//template <typename T>
//struct is_backward_impl : std::false_type {};
//template <>
//struct is_backward_impl<backward_tag> : std::false_type {};
//template <typename T>
//static constexpr auto is_backward = is_backward_impl<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
namespace tatooine::execution_policy {
//==============================================================================
struct parallel_t {};
static constexpr parallel_t parallel;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
struct sequential_t {};
static constexpr sequential_t sequential;
template <typename T>
concept policy = std::same_as<T, parallel_t> || std::same_as<T, sequential_t>;
//==============================================================================
}  // namespace tatooine::execution_policy
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
concept execution_policy_tag = same_as<T, execution_policy::sequential_t> ||
    same_as<T, execution_policy::parallel_t>;
//==============================================================================
}  // namespace tatooine
namespace tatooine::tag {
//==============================================================================
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
fill(Real) -> fill<Real>;

struct zeros_t {};
static constexpr zeros_t zeros;

struct ones_t {};
static constexpr ones_t ones;
//==============================================================================
}  // namespace tatooine::tag
//==============================================================================
#endif
