#ifndef TATOOINE_TYPE_TRAITS_H
#define TATOOINE_TYPE_TRAITS_H
//==============================================================================
#include <type_traits>
#include <array>
#include <complex>
#include <tatooine/void_t.h>
#include <tatooine/common_type.h>
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
static constexpr auto is_floating_point = (std::is_floating_point<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename ... Ts>
static constexpr auto is_arithmetic = (std::is_arithmetic<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename... Ts>
static constexpr auto is_integral = (std::is_integral<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename ... Ts>
static constexpr auto is_const = (std::is_const<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename ... Ts>
static constexpr auto is_non_const = (!std::is_const<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename ... Ts>
static constexpr auto is_signed = (std::is_signed<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename ... Ts>
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
    is_convertible<From, bool>           ||
    is_convertible<From, char>           ||
    is_convertible<From, unsigned char>  ||
    is_convertible<From, char16_t>       ||
    is_convertible<From, char32_t>       ||
    is_convertible<From, wchar_t>        ||
    is_convertible<From, unsigned short> ||
    is_convertible<From, short>          ||
    is_convertible<From, int>            ||
    is_convertible<From, unsigned int>   ||
    is_convertible<From, long>           ||
    is_convertible<From, unsigned long>;
//------------------------------------------------------------------------------
template <typename From>
static constexpr auto is_convertible_to_floating_point =
    is_convertible<From, float>  ||
    is_convertible<From, double> ||
    is_convertible<From, long double>;
//------------------------------------------------------------------------------
template <typename T>
struct is_complex_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename ... Ts>
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
template <typename ... Ts>
static constexpr auto is_range = (is_range_impl<Ts>::value && ...);
//------------------------------------------------------------------------------
template <typename T, typename = void>
struct is_indexable_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
struct is_indexable_impl<T,
                         void_t<decltype(std::declval<T>().at(size_t{})),
                                     decltype(std::declval<T>()[size_t{}])>>
    : std::true_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename... Ts>
static constexpr auto is_indexable = (is_indexable_impl<Ts>::value && ...);
//------------------------------------------------------------------------------
//template <typename T, typename = void>
//struct is_indexable_space_impl: std::false_type{};
//template <typename T>
//struct is_indexable_space_impl<
//  std::decay_t<T>::iterator,
//  decltype(std::declval<T>().at(size_t{})),
//  decltype(std::declval<T>()[size_t{}]),
//  decltype(std::declval<T>().size()),
//    { t.size() } -> convertible_to_integral;
//    { size(t)  } -> convertible_to_integral;
//    { t.front()  } -> convertible_to_floating_point;
//    { t.back()  } -> convertible_to_floating_point;
//    { t.begin()  } -> forward_iterator;
//    { begin(t)  } -> forward_iterator;
//    { t.end()  } -> forward_iterator;
//    { end(t)  } -> forward_iterator;
//>: std::true_type{};
//template <typename T>
//static constexpr is_indexable_space = is_indexable_space_impl<T>::value
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
template <typename ...Ts>
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
