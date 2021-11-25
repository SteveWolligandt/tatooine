#ifndef TATOOINE_VARIADIC_HELPERS_H
#define TATOOINE_VARIADIC_HELPERS_H
//==============================================================================
#include <cstdint>
//==============================================================================
namespace tatooine::variadic {
//==============================================================================
template <typename T, typename... Ts>
struct front_impl {
  using type = T;
};
//------------------------------------------------------------------------------
template <typename... Ts>
using front = typename front_impl<Ts...>::type;
//==============================================================================
template <typename... T>
struct back_impl;
//------------------------------------------------------------------------------
template <typename T>
struct back_impl<T> {
  using type = T;
};
//------------------------------------------------------------------------------
template <typename T, typename... Ts>
struct back_impl<T, Ts...> {
  using type = typename back_impl<Ts...>::type;
};
//------------------------------------------------------------------------------
template <typename... Ts>
using back = typename back_impl<Ts...>::type;
//==============================================================================
template <std::size_t I, std::size_t CurNum, std::size_t... RestNums>
struct ith_num_impl {
  static auto constexpr value = ith_num_impl<I - 1, RestNums...>::value;
};
template <std::size_t CurNum, std::size_t... RestNums>
struct ith_num_impl<0, CurNum, RestNums...> {
  static auto constexpr value = CurNum;
};
template <std::size_t I, std::size_t... Nums>
[[maybe_unused]] static auto constexpr ith_num =
    ith_num_impl<I, Nums...>::value;
//==============================================================================
template <std::size_t I, typename CurType, typename... RestTypes>
struct ith_type_impl {
  using type = typename ith_type_impl<I - 1, RestTypes...>::type;
};
template <typename CurType, typename... RestTypes>
struct ith_type_impl<0, CurType, RestTypes...> {
  using type = CurType;
};
template <std::size_t I, typename... Types>
using ith_type = typename ith_type_impl<I, Types...>::type;
//==============================================================================
template <std::size_t X, std::size_t... Rest>
struct contains_impl;
template <std::size_t X, std::size_t I, std::size_t... Rest>
struct contains_impl<X, I, Rest...>
    : std::integral_constant<bool, contains_impl<X, Rest...>::value> {};
template <std::size_t X, std::size_t... Rest>
struct contains_impl<X, X, Rest...> : std::true_type {};
template <std::size_t X>
struct contains_impl<X> : std::false_type {};
template <std::size_t X, std::size_t... Is>
static constexpr auto contains = contains_impl<X, Is...>::value;
//==============================================================================
}  // namespace tatooine::variadic
//==============================================================================
#endif
