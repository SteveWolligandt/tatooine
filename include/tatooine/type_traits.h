#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H
//==============================================================================
#include <type_traits>
#include <complex>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename... Ts>
using common_type = std::common_type_t<Ts...>;
//==============================================================================
template <typename T, typename S>
static constexpr auto is_same = std::is_same_v<T, S>;
//==============================================================================
template <typename F, typename... Args>
static constexpr auto is_invocable = std::is_invocable_v<F, Args...>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename R, typename F, typename... Args>
static constexpr auto is_invocable_r = std::is_invocable_v<R, F, Args...>;
//==============================================================================
template <typename F, typename... Ts>
static constexpr auto is_predicate =
    std::is_same_v<bool, std::invoke_result_t<F, Ts...>>;
//==============================================================================
template <typename T>
static constexpr auto is_floating_point = std::is_floating_point_v<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_arithmetic = std::is_arithmetic_v<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_integral = std::is_integral_v<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_const = std::is_const_v<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_non_const = !std::is_const_v<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_signed = std::is_signed_v<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_unsigned = std::is_unsigned_v<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_signed_integral = is_signed<T>&& is_integral<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_unsigned_integral = is_unsigned<T>&& is_integral<T>;
//------------------------------------------------------------------------------
template <typename From, typename To>
static constexpr auto is_convertible = std::is_convertible_v<From, To>;
//------------------------------------------------------------------------------
template <typename T>
struct is_complex_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_complex = is_complex_impl<T>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_complex_impl<std::complex<T>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_range_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_range_impl<T, std::void_t<decltype(std::declval<T>().begin()),
                                    decltype(std::declval<T>().end())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_range = is_range_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_indexable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_indexable_impl<T, std::void_t<decltype(std::declval<T>().at(size_t{})),
                                    decltype(std::declval<T>()[size_t{}])>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_indexable = is_indexable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_dereferencable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_dereferencable_impl<T, std::void_t<decltype(*std::declval<T>())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_dereferencable = is_dereferencable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_post_incrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_incrementable_impl<T, std::void_t<decltype(std::declval<T>()++)>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_post_incrementable =
    is_post_incrementable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_pre_incrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_incrementable_impl<T, std::void_t<decltype(++std::declval<T>())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_pre_incrementable =
    is_pre_incrementable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_post_decrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_decrementable_impl<T, std::void_t<decltype(std::declval<T>()--)>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_post_decrementable =
    is_post_decrementable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_pre_decrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_decrementable_impl<T, std::void_t<decltype(--std::declval<T>())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_pre_decrementable =
    is_pre_decrementable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename S, typename = void>
struct are_equality_comparable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename S>
struct are_equality_comparable_impl<
    T, S, std::void_t<decltype(std::declval<T>() == std::declval<S>())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T, typename S>
static constexpr auto are_equality_comparable =
    are_equality_comparable_impl<T, S>::value;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_forward_iterator = are_equality_comparable<T, T>&&
    is_pre_incrementable<T>&& is_dereferencable<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_backward_iterator = are_equality_comparable<T, T>&&
    is_pre_decrementable<T>&& is_dereferencable<T>;
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto               is_bidirectional_iterator =
    are_equality_comparable<T, T>&& is_pre_incrementable<T>&&
        is_pre_decrementable<T>&& is_dereferencable<T>;
//==============================================================================
template <bool... Preds>
using enable_if = std::enable_if_t<(Preds && ...), bool>;
//------------------------------------------------------------------------------
template <typename T, typename S>
using enable_if_same = enable_if<is_same<T, S>>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_floating_point = enable_if<is_floating_point<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_arithmetic = enable_if<is_arithmetic<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_integral = enable_if<is_integral<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_signed_integral = enable_if<is_signed_integral<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_unsigned_integral = enable_if<is_unsigned_integral<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_complex = enable_if<is_complex<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_arithmetic_or_complex =
    enable_if<(is_arithmetic<Ts> || is_complex<Ts>)...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_range = enable_if<is_range<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_indexable = enable_if<is_indexable<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_forward_iterator = enable_if<is_forward_iterator<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_backward_iterator = enable_if<is_backward_iterator<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_bidirectional_iterator =
    enable_if<is_bidirectional_iterator<Ts>...>;
//------------------------------------------------------------------------------
template <typename From, typename To>
using enable_if_convertible = enable_if<is_convertible<From, To>>;
//------------------------------------------------------------------------------
template <typename F, typename... Args>
using enable_if_invocable = enable_if<is_invocable<F, Args...>>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_const = enable_if<is_const<Ts>...>;
//------------------------------------------------------------------------------
template <typename... Ts>
using enable_if_non_const = enable_if<is_non_const<Ts>...>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
