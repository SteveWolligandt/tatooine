#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/common_type.h>
#include <tatooine/void_t.h>

#include <array>
#include <complex>
#include <type_traits>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, std::size_t = sizeof(T)>
auto type_exists_impl(T*) -> std::true_type;
auto type_exists_impl(...) -> std::false_type;

template <typename T>
static constexpr auto type_exists =
    decltype(type_exists_impl(std::declval<T*>()))::value;
//==============================================================================
template <typename T>
static constexpr auto is_pointer = std::is_pointer<T>::value;
//==============================================================================
template <typename T>
struct is_array_impl : std::false_type {};
template <typename T, std::size_t N>
struct is_array_impl<std::array<T, N>> : std::true_type {};
template <typename T>
static auto constexpr is_array = is_array_impl<T>::value;
//==============================================================================
template <typename... Ts>
struct is_same_impl;
template <typename T0, typename T1, typename T2, typename... Ts>
struct is_same_impl<T0, T1, T2, Ts...>
    : std::integral_constant<bool, is_same_impl<T0, T1>::value &&
                                       is_same_impl<T1, T2, Ts...>::value> {};
template <typename T0, typename T1>
struct is_same_impl<T0, T1> : std::is_same<T0, T1> {};
template <typename T0>
struct is_same_impl<T0> : std::true_type {};
template <typename... Ts>
static constexpr auto is_same = is_same_impl<Ts...>::value;
//==============================================================================
template <typename... Ts>
static constexpr auto is_float = (is_same<Ts, float> && ...);
//==============================================================================
template <typename... Ts>
static constexpr auto is_double = (is_same<Ts, double> && ...);
//==============================================================================
template <typename... Ts>
static constexpr auto is_int = (is_same<Ts, int> && ...);
//==============================================================================
template <typename... Ts>
static constexpr auto is_size_t = (is_same<Ts, size_t> && ...);
//==============================================================================
template <typename Query, typename... Others>
static constexpr auto is_either_of = (is_same<Query, Others> || ...);
//==============================================================================
template <typename F, typename... Args>
static constexpr auto is_invocable = std::is_invocable<F, Args...>::value;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename R, typename F, typename... Args>
static constexpr auto is_invocable_r = std::is_invocable<R, F, Args...>::value;
//==============================================================================
template <typename F, typename... Ts>
static constexpr auto is_predicate =
    is_same<bool, std::invoke_result_t<F, Ts...>>;
//==============================================================================
template <typename... Ts>
static constexpr auto is_floating_point = (std::is_floating_point<Ts>::value &&
                                           ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_arithmetic = (std::is_arithmetic<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_integral = (std::is_integral<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_const = (std::is_const<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_non_const = (!std::is_const<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_signed = (std::is_signed<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_unsigned = (std::is_unsigned<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_signed_integral = ((is_signed<Ts> && ...) &&
                                            (is_integral<Ts> && ...));
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_unsigned_integral = ((is_unsigned<Ts> && ...) &&
                                              (is_integral<Ts> && ...));
//------------------------------------------------------------------------------
template <typename From, typename To>
static constexpr auto is_convertible = std::is_convertible<From, To>::value;
//------------------------------------------------------------------------------
template <typename From>
static constexpr auto is_convertible_to_integral =
    is_convertible<From, bool> || is_convertible<From, char> ||
    is_convertible<From, unsigned char> || is_convertible<From, char16_t> ||
    is_convertible<From, char32_t> || is_convertible<From, wchar_t> ||
    is_convertible<From, unsigned short> || is_convertible<From, short> ||
    is_convertible<From, int> || is_convertible<From, unsigned int> ||
    is_convertible<From, long> || is_convertible<From, unsigned long>;
//------------------------------------------------------------------------------
template <typename From>
static constexpr auto is_convertible_to_floating_point =
    is_convertible<From, float> || is_convertible<From, double> ||
    is_convertible<From, long double>;
//------------------------------------------------------------------------------
template <typename T>
struct is_complex_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto is_complex = (is_complex_impl<Ts>::value && ...);
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto is_arithmetic_or_complex = ((is_arithmetic<Ts> ||
                                                   is_complex<Ts>)&&...);
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_complex_impl<std::complex<T>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_range_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_range_impl<T, void_t<decltype(std::declval<T>().begin()),
                               decltype(std::declval<T>().end())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto is_range = (is_range_impl<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_indexable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_indexable_impl<T, void_t<decltype(std::declval<T>().at(size_t{})),
                                   decltype(std::declval<T>()[size_t{}])>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto is_indexable = (is_indexable_impl<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_dereferencable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_dereferencable_impl<T*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_dereferencable_impl<T, void_t<decltype(*std::declval<T>())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto is_dereferencable = (is_dereferencable_impl<Ts>::value &&
                                           ...);
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_post_incrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_incrementable_impl<T, void_t<decltype(std::declval<T>()++)>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_incrementable_impl<T*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_incrementable_impl<T const*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto is_post_incrementable =
    (is_post_incrementable_impl<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_pre_incrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_incrementable_impl<T, void_t<decltype(++std::declval<T>())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_incrementable_impl<T*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_incrementable_impl<T const*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_pre_incrementable =
    is_pre_incrementable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_post_decrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_decrementable_impl<T, void_t<decltype(std::declval<T>()--)>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_decrementable_impl<T*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_post_decrementable_impl<T const*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_post_decrementable =
    is_post_decrementable_impl<T>::value;
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_pre_decrementable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_decrementable_impl<T, void_t<decltype(--std::declval<T>())>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_decrementable_impl<T*> : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_pre_decrementable_impl<T const*> : std::true_type {};
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
    T, S, void_t<decltype(std::declval<T>() == std::declval<S>())>>
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
template <typename T>
struct value_type_impl;
//------------------------------------------------------------------------------
template <typename T>
requires requires { typename T::value_type; }
struct value_type_impl<T> {
  using type = typename T::value_type;
};
//------------------------------------------------------------------------------
template <typename T>
requires is_arithmetic<T>
struct value_type_impl<T> {
  using type = T;
};
//------------------------------------------------------------------------------
template <typename T>
using value_type = typename value_type_impl<T>::type;
//==============================================================================
template <typename T>
struct is_pair_impl : std::false_type {};
//------------------------------------------------------------------------------
template <typename First, typename Second>
struct is_pair_impl<std::pair<First, Second>> : std::true_type {};
//------------------------------------------------------------------------------
template <typename T>
static constexpr auto is_pair = is_pair_impl<T>::value;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
