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
template <typename F, typename... Ts>
using is_predicate_impl = std::is_same<bool, std::invoke_result_t<F, Ts...>>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename... Ts>
static constexpr inline auto is_predicate = is_predicate_impl<F, Ts...>::value;
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
template <typename From, typename To>
using enable_if_convertible = enable_if<is_convertible<From, To>>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
